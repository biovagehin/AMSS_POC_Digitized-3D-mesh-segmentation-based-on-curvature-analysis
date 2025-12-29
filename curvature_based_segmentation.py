#### Curvature-Based Segmentation

## STEP 0 : Libraries

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from math import cos, sin, atan2
import time

#----------------------------------------------------------------------------------------------------------
## STEP 1 : Load STL File, Extract Vertices, their Normals, their 2-Ring Neighbours and the Mean Edge Length
file_path = input("Enter the path to your STL file: ").strip()
stl_mesh = mesh.Mesh.from_file(file_path)
# Visualization of the mesh
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
triangles = stl_mesh.vectors
ax.add_collection3d(Poly3DCollection(triangles, facecolor='lightblue', edgecolor='k', alpha=0.5))
points = triangles.reshape(-1, 3)
ax.set_xlim(points[:,0].min(), points[:,0].max())
ax.set_ylim(points[:,1].min(), points[:,1].max())
ax.set_zlim(points[:,2].min(), points[:,2].max())
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#ax.auto_scale_xyz(points[:,0], points[:,1], points[:,2])
plt.show()
'''

# extract vertices, their normals, their 2-ring neighbours, and the mean edge length
def extract_mesh_data(stl_mesh):
    points = stl_mesh.vectors.reshape(-1, 3)
    face_normals = np.repeat(stl_mesh.normals, 3, axis=0)

    vertices = []  # list of unique vertices
    normals = []   # n = (Σ adjacent face normals)/||Σ adjacent face normals||
    index_map = {} # to map vertex coordinates to their index in vertices list

    for i, v in enumerate(points):
        key = tuple(v)
        if key not in index_map:
            index_map[key] = len(vertices)
            vertices.append(points[i])
            normals.append(face_normals[i])
        else:
            idx = index_map[key]
            normals[idx] = normals[idx] + face_normals[i]

    for i in range(len(normals)):
        normals[i] = normals[i] / np.linalg.norm(normals[i]) # normalisation of normals

    # 1-ring neighbours search + mean edge length calculation
    neighbours_1r = [[] for _ in range(len(vertices))]
    triangles = stl_mesh.vectors.reshape(-1, 3, 3) 
    mean_edge_lengths = 0
    n_edges = 0
    for tri in triangles:
        idx0 = index_map[tuple(tri[0])]
        idx1 = index_map[tuple(tri[1])]
        idx2 = index_map[tuple(tri[2])]

        if idx1 not in neighbours_1r[idx0]:
            neighbours_1r[idx0].append(idx1)
        if idx2 not in neighbours_1r[idx0]:
            neighbours_1r[idx0].append(idx2)

        if idx0 not in neighbours_1r[idx1]:
            neighbours_1r[idx1].append(idx0)
        if idx2 not in neighbours_1r[idx1]:
            neighbours_1r[idx1].append(idx2)

        if idx0 not in neighbours_1r[idx2]:
            neighbours_1r[idx2].append(idx0)
        if idx1 not in neighbours_1r[idx2]:
            neighbours_1r[idx2].append(idx1)

        # edge lengths
        mean_edge_lengths += np.linalg.norm(tri[0] - tri[1])
        mean_edge_lengths += np.linalg.norm(tri[1] - tri[2])
        mean_edge_lengths += np.linalg.norm(tri[2] - tri[0])
        n_edges += 3
    mean_edge_lengths /= n_edges 

    # 2-ring neighbours search - they are the neighbours of the 1-ring neighbours
    neighbours_2r = []
    for i in range(len(vertices)):
        two_ring = set(neighbours_1r[i])
        for j in neighbours_1r[i]:
            two_ring.update(neighbours_1r[j])
        two_ring.discard(i)  # to remove the vertex itself
        neighbours_2r.append(list(two_ring))

    return vertices, normals, neighbours_2r, mean_edge_lengths, index_map

#----------------------------------------------------------------------------------------------------------
## STEP 2 : Compute Local Curvature at Each Vertex of the Mesh
# Compute local curvature at a vertex P given its normal N and its 2-ring neighbours
def local_curvature(vertex, normal, neighbors, neighbor_normals):
    # curvature at P in the direction t_i defined by neighbor i :
    # k_i(t_i) = <(P_i - P), (N_i - N)> / ||P_i - P||^2
    # to approximate the principal curvatures k_1 and k_2, we use the relation :
    # k(phi) = k_1 * cos^2(phi) + k_2 * sin^2(phi)
    # -> phi : angle between t_i and a reference direction t_1
    
    P = vertex
    N = normal

    ki_list = [] 
    ti_list = [] 
    phii_list = [] 

    if len(neighbors) < 3:
        return 0.0  # not enough neighbors to compute curvature

    for i in range(len(neighbors)):
        Pi = neighbors[i]
        Ni = neighbor_normals[i]
        di = Pi - P
        ti = (di - np.dot(di, N) * N) / np.linalg.norm(di - np.dot(di, N) * N)
        ki = (np.dot(di, Ni - N)) / (np.linalg.norm(di)**2)

        if len(ti_list) == 0:
            phii = 0.0
        else:
            ref = ti_list[0]
            phii = atan2(np.dot(ti, np.cross(N, ref)), np.dot(ti, ref))


        ki_list.append(ki)
        ti_list.append(ti)
        phii_list.append(phii)

    # Curve fitting to find k_max, k_min, phi_max
    def curvature_model(phi, k1, k2, phi1):
        return k1 * np.cos(phi - phi1)**2 + k2 * np.sin(phi - phi1)**2
    try:
        res, _ = curve_fit(
            curvature_model,
            phii_list,
            ki_list,
            p0=[max(ki_list), min(ki_list), 0],
            maxfev=5000
        )
        k_max, k_min, phi_max = res
    except Exception: # if curve fitting fails
        k_max = max(ki_list)

    return k_max

# Compute curvatures for all vertices
def compute_curvatures(stl_mesh):
    vertices, normals, neighbours, mean_edge_lengths, index_map = extract_mesh_data(stl_mesh)
    normalised_curvatures = []
    for i in range(len(vertices)):
        P = vertices[i]
        N = normals[i]
        neighbors = [vertices[j] for j in neighbours[i]]
        neighbor_normals = [normals[j] for j in neighbours[i]]

        k = local_curvature(P, N, neighbors, neighbor_normals)
        normalised_curvatures.append(k*mean_edge_lengths) # normalisation with the mean edge length

    return vertices, normalised_curvatures, index_map

# Visualization of curvatures on the 3D mesh 
# triangles are colored according to the mean curvature of their vertices
def plot_curvature_3d_triangles(stl_mesh):
    vertices, curvatures, index_map = compute_curvatures(stl_mesh)
    v = np.array(vertices)
    k = np.array(curvatures)
    # mean curvature per triangle
    triangles = stl_mesh.vectors
    triangle_curvatures = np.array([
        np.mean([k[index_map[tuple(tri[i])]] for i in range(3)])
        for tri in triangles
    ])

    # plot
    cmap = plt.get_cmap("jet")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = cmap(triangle_curvatures)
    poly_collection = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1)
    ax.add_collection3d(poly_collection)

    ax.set_xlim(v[:,0].min(), v[:,0].max())
    ax.set_ylim(v[:,1].min(), v[:,1].max())
    ax.set_zlim(v[:,2].min(), v[:,2].max())

    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(triangle_curvatures)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label("Principal curvature normalised (unitless)")

    plt.show()
# vertices are colored according to their curvature
def plot_curvature_3d_vertices(stl_mesh):
    # Compute of curvatures at vertices
    vertices, curvatures, _ = compute_curvatures(stl_mesh)
    v = np.array(vertices)
    k = np.array(curvatures)
    # plot
    cmap = plt.get_cmap("jet")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Vertices coloring according to curvature values
    colors = cmap((k - k.min()) / (k.max() - k.min()))  # normalisation between 0 and 1
    ax.scatter(v[:,0], v[:,1], v[:,2], c=colors, s=10) 
    ax.set_xlim(v[:,0].min(), v[:,0].max())
    ax.set_ylim(v[:,1].min(), v[:,1].max())
    ax.set_zlim(v[:,2].min(), v[:,2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(k)
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label("Principal curvature normalised (unitless)")

    plt.show()

#----------------------------------------------------------------------------------------------------------
## STEP 3 : Analysis of the distribution of curvatures
# Distribution of normalised curvatures (k_max * mean_edge_length) -> discrete histogram
def plot_curvature_distribution(curvatures, bins=50, filename="results/distibution_discrete.png"):
    plt.figure(figsize=(10,6))
    plt.hist(curvatures, bins=bins, color='grey', edgecolor='black')
    plt.title("Distribution of Principal Curvatures (Normalised)")
    plt.xlabel(f"Principal curvature normalised")
    plt.ylabel("Number of occurrences")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename, dpi=300)
    #plt.show()

# Estimation of curvature distribution using kernel density estimation -> continuous curve
def kernel_estimation(curvatures, n_bins=50, h=0.01, filename="results/distribution_continuous.png"):
    curvatures = np.array(curvatures)

    #plot_curvature_distribution(curvatures, bins=n_bins, filename=f"results/distribution_discrete_n{n_bins}_h{h}.png")
    
    # discrete distribution over bins
    bin_edges = np.linspace(curvatures.min(), curvatures.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # computation of density estimation using Gaussian kernels
    density = np.zeros_like(bin_centers)
    for xi in curvatures:
        density += np.exp(-0.5 * ((bin_centers - xi)/h)**2) / (h * np.sqrt(2*np.pi))
    density /= density.sum() * (bin_centers[1] - bin_centers[0])

    # visualization
    """ plt.figure(figsize=(10,6))
    plt.plot(bin_centers, density, color='black')
    plt.title("Estimation of Curvature Distribution using KDE")
    plt.xlabel("Principal curvature normalised")
    plt.ylabel("Density")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(filename, dpi=300)
    plt.show() """
    
    return bin_centers, density

# Analysis of the distribution : identification of peaks and valleys
def analyze_distribution(curvatures, bin_centers, density, window_size=10, n_bins=50, h=0.01, min_density=0.02):
    n_bins = len(bin_centers)

    # Robust peaks detection : maximum in a window
    peaks = []
    for i in range(n_bins):
        start = max(0, i - window_size // 2)
        end = min(n_bins, i + window_size // 2 + 1)
        if density[i] == np.max(density[start:end]):
            peaks.append(i)

    # Filtering peaks with minimum height (0.01% of max density)
    min_peak_height = 0.01 * density.max()
    peaks = [p for p in peaks if density[p] >= min_peak_height]

    if len(peaks) < 1:
        return [], []

    # Addition of artificial peaks for extremities valleys detection
    peaks_ext = []
    density_ext = density.copy()

    # Left Extremity
    left_data = bin_centers[:peaks[0]]
    if len(left_data) > 1:
        mu_l = np.mean(left_data)
        sigma_l = np.std(left_data)
        x_ext_l = mu_l - 3 * sigma_l
    else:
        x_ext_l = bin_centers[0]
    idx_ext_l = np.argmin(np.abs(bin_centers - x_ext_l))
    peaks_ext.append(idx_ext_l)
    peaks_ext.extend(peaks)
    density_ext[idx_ext_l] = 0


    # Right Extremity
    right_data = bin_centers[peaks[-1]:]
    if len(right_data) > 1:
        mu_r = np.mean(right_data)
        sigma_r = np.std(right_data)
        x_ext_r = mu_r + 3 * sigma_r
    else:
        x_ext_r = bin_centers[-1]
    idx_ext_r = np.argmin(np.abs(bin_centers - x_ext_r))
    peaks_ext.append(idx_ext_r)
    density_ext[idx_ext_r] = 0


    # Robust valleys detection : farthest bin from the line connecting two consecutive peaks
    valleys = []
    for p1, p2 in zip(peaks_ext[:-1], peaks_ext[1:]):
        # line between both peaks : y = m*x + b
        x1, y1 = bin_centers[p1], density_ext[p1]
        x2, y2 = bin_centers[p2], density_ext[p2]
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        b = y1 - m * x1
        start = min(p1, p2) + 1
        end = max(p1, p2)
        if start >= end:
            continue

        distances = []
        for i in range(start, end):
            y_line = m * bin_centers[i] + b
            distances.append(y_line - density[i])
        distances = np.array(distances)
        
        # if several candidates, keep the one closest to 0
        if len(distances) > 0:
            max_dist_idx = np.argmax(distances)
            candidates = np.where(distances == distances[max_dist_idx])[0]
            best_idx = candidates[np.argmin(np.abs(bin_centers[start:end][candidates]))]
            valleys.append(start + best_idx)

    # Filtering extreme valleys if their density is too low
    filtered_valleys = []
    ignored_valleys = []

    if len(valleys) > 0:
        # Left extreme valley
        if density[valleys[0]] >= min_density * density.max():
            filtered_valleys.append(valleys[0])
        else:
            ignored_valleys.append(valleys[0])

    # All intermediate valleys are always kept
    for v in valleys[1:-1]:
        filtered_valleys.append(v)

    if len(valleys) > 1:
        # Right extreme valley
        if density[valleys[-1]] >= min_density * density.max():
            filtered_valleys.append(valleys[-1])
        else:
            ignored_valleys.append(valleys[-1])



    # Visualization of peaks and valleys
    """ plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, density, color="black", label="Distribution")

    plt.scatter(bin_centers[peaks], density[peaks], color="red", label="Peaks", zorder=3)
    plt.scatter(bin_centers[filtered_valleys], density[filtered_valleys], color="blue", label="Valleys kept", zorder=3)
    ignored_valleys = [v for v in valleys if v not in filtered_valleys]
    plt.scatter(bin_centers[ignored_valleys], density[ignored_valleys], color="grey", label="Valleys ignored", zorder=3)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"results/distribution_analysis_n{n_bins}_h{h}.png", dpi=300)
    #plt.show() """

    # Visualization of discrete and continuous distributions with peaks and valleys superimposed
    plt.figure(figsize=(10, 8))
    plt.hist(
        curvatures,
        bins=n_bins,
        density=True,
        color="lightgrey",
        edgecolor="black",
        alpha=0.4,
        label="Discrete distribution",
    )
    plt.plot(bin_centers, density, color="black", linewidth=2, label="KDE")
    plt.scatter(
        bin_centers[peaks],
        density[peaks],
        color="red",
        s=60,
        label="Peaks",
        zorder=3
    )
    plt.scatter(
        bin_centers[filtered_valleys],
        density[filtered_valleys],
        color="blue",
        s=60,
        label="Keptvalleys",
        zorder=3
    )
    ignored_valleys = [v for v in valleys if v not in filtered_valleys]
    if len(ignored_valleys) > 0:
        plt.scatter(
            bin_centers[ignored_valleys],
            density[ignored_valleys],
            color="grey",
            s=60,
            label="Ignored valleys",
            zorder=3
        )
    plt.xlabel("Principal curvature normalised")
    plt.ylabel("Density")
    plt.title("Distribution of Principal Curvatures with Peaks and Valleys - Discrete and Continuous")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(f"results/distribution_discrete_analysis_n{n_bins}_h{h}.png", dpi=300)
    #plt.show()

    return peaks, filtered_valleys, n_bins, h

#----------------------------------------------------------------------------------------------------------
## STEP 4 : Segmentation of the Mesh Based on Curvature Thresholds
# Automatic definition of curvature thresholds based on the distribution analysis, corresponding to the first and last valleys
def thresholding(vertices, curvatures, n_bins=50, h=0.01, window_size=10, min_density=0.02):
    vertices = np.array(vertices)
    curvatures = np.array(curvatures)

    # Kernel estimation of the curvature distribution
    bin_centers, density = kernel_estimation(curvatures, n_bins=n_bins, h=h)
    # Detection of peaks and valleys
    peaks, valleys, n_bins, h = analyze_distribution(curvatures, bin_centers, density, window_size=window_size, n_bins=n_bins, h=h, min_density=min_density)
    if len(valleys) < 2:
        raise ValueError("Not enough valleys.")

    # Thresholds based on the first and last valleys
    first_valley_idx = valleys[0] # we keep the 1st valley
    last_valley_idx = valleys[-1] # and the last valley
    threshold_low = bin_centers[first_valley_idx]
    threshold_high = bin_centers[last_valley_idx]

    # Vertices labeling
    labels = []
    for k in curvatures:
        if threshold_low <= k <= threshold_high:
            labels.append("uniform")
        else:
            labels.append("edge")
    labels = np.array(labels)

    #print(f"Thresholds : {threshold_low:.4f} and {threshold_high:.4f}")

    # Visualization of the mesh after thresholding
    """ colors = np.array(['blue' if label == 'uniform' else 'red' for label in labels])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c=colors, s=5)
    ax.set_xlim(vertices[:,0].min(), vertices[:,0].max())
    ax.set_ylim(vertices[:,1].min(), vertices[:,1].max())
    ax.set_zlim(vertices[:,2].min(), vertices[:,2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(" Classification of vertices : uniform (blue) vs edge (red)")
    plt.show()"""

    return labels, threshold_low, threshold_high 

# Segmentation function, using a region growing algorithm
def segmentation(stl_mesh, n_bins=50, h=0.01, window_size=10, min_density=0.02, n_erosions=5):
    start_time = time.perf_counter()
    # Compute curvatures
    vertices, curvatures, index_map = compute_curvatures(stl_mesh)
    
    # Thresholding to label vertices as 'uniform' or 'edge'
    labels, threshold_low, threshold_high = thresholding(vertices, curvatures, n_bins=n_bins, h=h, window_size=window_size, min_density=min_density)

    triangles = stl_mesh.vectors
    n_triangles = len(triangles)

    def triangle_area(tri):
        return 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    triangle_areas = np.array([triangle_area(tri) for tri in triangles])
    mean_triangle_area = np.mean(triangle_areas)
    area_threshold = 20 * mean_triangle_area 

    # Labelling triangles
    triangle_labels = []
    for tri in triangles:
        indices = [index_map[tuple(v)] for v in tri]
        tri_vertex_labels = [labels[idx] for idx in indices]
        if tri_vertex_labels.count("edge") >= 2:
            triangle_labels.append("edge")
        else:
            triangle_labels.append("uniform")
    triangle_labels = np.array(triangle_labels)

    # Triangle adjacency
    triangle_neighbors = [[] for _ in range(n_triangles)]
    vertex_to_triangles = {i: [] for i in range(len(vertices))}
    for t_idx, tri in enumerate(triangles):
        for v in tri:
            vertex_to_triangles[index_map[tuple(v)]].append(t_idx)
    for t_idx, tri in enumerate(triangles):
        neighbor_set = set()
        for v in tri:
            neighbor_set.update(vertex_to_triangles[index_map[tuple(v)]])
        neighbor_set.discard(t_idx)
        triangle_neighbors[t_idx] = list(neighbor_set)

    # Initial region growing on uniform triangles
    region_ids = -1 * np.ones(n_triangles, dtype=int)
    current_region = 0
    for t_idx in range(n_triangles):
        if triangle_labels[t_idx] != "uniform" or region_ids[t_idx] != -1:
            continue
        stack = [t_idx]
        while stack:
            curr = stack.pop()
            if region_ids[curr] != -1:
                continue
            region_ids[curr] = current_region
            for nbr in triangle_neighbors[curr]:
                if triangle_labels[nbr] == "uniform" and region_ids[nbr] == -1:
                    stack.append(nbr)
        current_region += 1

    # Post-processing small regions
    region_ids_post = region_ids.copy()
    unique_regions = np.unique(region_ids[region_ids >= 0])
    for r in unique_regions:
        tris = np.where(region_ids == r)[0]
        area = triangle_areas[tris].sum()
        if area < area_threshold:
            neighbors = set()
            for t in tris:
                for n in triangle_neighbors[t]:
                    if region_ids[n] != r and region_ids[n] != -1:
                        neighbors.add(region_ids[n])
            if len(neighbors) == 1:
                target = neighbors.pop()
                region_ids_post[tris] = target
            else:
                region_ids_post[tris] = -1
    region_ids = region_ids_post
    current_region = len(np.unique(region_ids[region_ids >= 0]))
    print(f"Number of regions after post-processing (before erosion/dilation) : {current_region}")

    # Visualization of the initial segmentation
    """ cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))  # edges
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly_collection = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.7)
    ax.add_collection3d(poly_collection)
    v = np.array(vertices)
    ax.set_xlim(v[:,0].min(), v[:,0].max())
    ax.set_ylim(v[:,1].min(), v[:,1].max())
    ax.set_zlim(v[:,2].min(), v[:,2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Initial Segmentation")
    plt.show() """

    ### Post-processing to enhance the segmentation : Erosion/Dilatation
    # Erosion
    mask = (region_ids != -1)
    for _ in range(n_erosions):
        new_mask = mask.copy()
        for t in range(n_triangles):
            if not mask[t]:
                continue
            same = sum(mask[n] for n in triangle_neighbors[t])
            if same < 3:
                new_mask[t] = False
        mask = new_mask

    # Region growing on eroded regions
    region_ids = -1 * np.ones(n_triangles, dtype=int)
    current_region = 0
    for t_idx in range(n_triangles):
        if not mask[t_idx] or region_ids[t_idx] != -1:
            continue
        stack = [t_idx]
        while stack:
            curr = stack.pop()
            if region_ids[curr] != -1:
                continue
            region_ids[curr] = current_region
            for nbr in triangle_neighbors[curr]:
                if mask[nbr] and region_ids[nbr] == -1:
                    stack.append(nbr)
        current_region += 1

    # Visualization after erosion + region growing
    """ cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.9)
    ax.add_collection3d(poly)
    v = np.array(vertices)
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Segmentation after erosion and second region growing")
    plt.tight_layout()
    plt.show() """

    # Dilatation
    for _ in range(n_erosions):
        new_region_ids = region_ids.copy()
        for t in range(n_triangles):
            if region_ids[t] != -1:
                continue
            neighbor_regions = [region_ids[n] for n in triangle_neighbors[t] if region_ids[n] != -1]
            if len(neighbor_regions) == 1:
                new_region_ids[t] = neighbor_regions[0]
        region_ids = new_region_ids

    n_regions_after = len(np.unique(region_ids[region_ids >= 0]))
    print(f"Number of regions after erosion/dilation : {n_regions_after}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Final visualization of the segmentation
    cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly_collection = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.7)
    ax.add_collection3d(poly_collection)
    v = np.array(vertices)
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Final Segmentation")
    plt.show()


    return region_ids, triangle_labels, threshold_low, threshold_high, elapsed_time

# alternative segmentation function with modified triangle labelling criterion : a triangle is labelled as 'edge' if it has at least one 'edge' vertex
def segmentation_bis(stl_mesh, n_bins=50, h=0.01, window_size=10, min_density=0.02, n_erosions=5):
    start_time = time.perf_counter()
    # Compute curvatures
    vertices, curvatures, index_map = compute_curvatures(stl_mesh)
    
    # Thresholding to label vertices as 'uniform' or 'edge'
    labels, threshold_low, threshold_high = thresholding(vertices, curvatures, n_bins=n_bins, h=h, window_size=window_size, min_density=min_density)

    triangles = stl_mesh.vectors
    n_triangles = len(triangles)

    def triangle_area(tri):
        return 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    triangle_areas = np.array([triangle_area(tri) for tri in triangles])
    mean_triangle_area = np.mean(triangle_areas)
    area_threshold = 20 * mean_triangle_area 

    # Labelling triangles
    triangle_labels = []
    for tri in triangles:
        indices = [index_map[tuple(v)] for v in tri]
        tri_vertex_labels = [labels[idx] for idx in indices]
        if tri_vertex_labels.count("edge") >= 1: ### MODIFIED HERE
            triangle_labels.append("edge")
        else:
            triangle_labels.append("uniform")
    triangle_labels = np.array(triangle_labels)

    # Triangle adjacency
    triangle_neighbors = [[] for _ in range(n_triangles)]
    vertex_to_triangles = {i: [] for i in range(len(vertices))}
    for t_idx, tri in enumerate(triangles):
        for v in tri:
            vertex_to_triangles[index_map[tuple(v)]].append(t_idx)
    for t_idx, tri in enumerate(triangles):
        neighbor_set = set()
        for v in tri:
            neighbor_set.update(vertex_to_triangles[index_map[tuple(v)]])
        neighbor_set.discard(t_idx)
        triangle_neighbors[t_idx] = list(neighbor_set)

    # Initial region growing on uniform triangles
    region_ids = -1 * np.ones(n_triangles, dtype=int)
    current_region = 0
    for t_idx in range(n_triangles):
        if triangle_labels[t_idx] != "uniform" or region_ids[t_idx] != -1:
            continue
        stack = [t_idx]
        while stack:
            curr = stack.pop()
            if region_ids[curr] != -1:
                continue
            region_ids[curr] = current_region
            for nbr in triangle_neighbors[curr]:
                if triangle_labels[nbr] == "uniform" and region_ids[nbr] == -1:
                    stack.append(nbr)
        current_region += 1

    # Post-processing small regions
    region_ids_post = region_ids.copy()
    unique_regions = np.unique(region_ids[region_ids >= 0])
    for r in unique_regions:
        tris = np.where(region_ids == r)[0]
        area = triangle_areas[tris].sum()
        if area < area_threshold:
            neighbors = set()
            for t in tris:
                for n in triangle_neighbors[t]:
                    if region_ids[n] != r and region_ids[n] != -1:
                        neighbors.add(region_ids[n])
            if len(neighbors) == 1:
                target = neighbors.pop()
                region_ids_post[tris] = target
            else:
                region_ids_post[tris] = -1
    region_ids = region_ids_post
    current_region = len(np.unique(region_ids[region_ids >= 0]))
    #print(f"Number of regions after post-processing (before erosion/dilation) : {current_region}")

    # Visualization of the initial segmentation
    """ cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))  # edges
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly_collection = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.7)
    ax.add_collection3d(poly_collection)
    v = np.array(vertices)
    ax.set_xlim(v[:,0].min(), v[:,0].max())
    ax.set_ylim(v[:,1].min(), v[:,1].max())
    ax.set_zlim(v[:,2].min(), v[:,2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Initial Segmentation")
    plt.show() """

    ### Post-processing to enhance the segmentation : Erosion/Dilatation
    # Erosion
    mask = (region_ids != -1)
    for _ in range(n_erosions):
        new_mask = mask.copy()
        for t in range(n_triangles):
            if not mask[t]:
                continue
            same = sum(mask[n] for n in triangle_neighbors[t])
            if same < 3:
                new_mask[t] = False
        mask = new_mask

    # Region growing on eroded regions
    region_ids = -1 * np.ones(n_triangles, dtype=int)
    current_region = 0
    for t_idx in range(n_triangles):
        if not mask[t_idx] or region_ids[t_idx] != -1:
            continue
        stack = [t_idx]
        while stack:
            curr = stack.pop()
            if region_ids[curr] != -1:
                continue
            region_ids[curr] = current_region
            for nbr in triangle_neighbors[curr]:
                if mask[nbr] and region_ids[nbr] == -1:
                    stack.append(nbr)
        current_region += 1

    # Visualization after erosion + region growing
    """ cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.9)
    ax.add_collection3d(poly)
    v = np.array(vertices)
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Segmentation after erosion and second region growing")
    plt.tight_layout()
    plt.show() """

    # Dilatation
    for _ in range(n_erosions):
        new_region_ids = region_ids.copy()
        for t in range(n_triangles):
            if region_ids[t] != -1:
                continue
            neighbor_regions = [region_ids[n] for n in triangle_neighbors[t] if region_ids[n] != -1]
            if len(neighbor_regions) == 1:
                new_region_ids[t] = neighbor_regions[0]
        region_ids = new_region_ids

    n_regions_after = len(np.unique(region_ids[region_ids >= 0]))
    #print(f"Number of regions after erosion/dilation : {n_regions_after}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Final visualization of the segmentation
    cmap = plt.get_cmap("tab20")
    colors = []
    for t in range(n_triangles):
        if region_ids[t] == -1:
            colors.append((0, 0, 0, 0.7))
        else:
            colors.append(cmap(region_ids[t] % 20))
    colors = np.array(colors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    poly_collection = Poly3DCollection(triangles, facecolors=colors, edgecolor='k', linewidths=0.1, alpha=0.7)
    ax.add_collection3d(poly_collection)
    v = np.array(vertices)
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title("Final Segmentation")
    plt.show()


    return region_ids, triangle_labels, threshold_low, threshold_high, elapsed_time

#----------------------------------------------------------------------------------------------------------
# STEP 5 : Evaluation of performance on an STL file

region_ids, _, t_low, t_high, execution_time = segmentation(stl_mesh, n_bins=200, h=0.01, window_size=8, min_density=0.02, n_erosions=6)

print("----- RESULTS -----------------------------------------------------------------")
print(f"Number of triangles in the mesh : {len(stl_mesh.vectors)}")
print(f"Execution time : {execution_time:.2f} seconds")
print(f"Thresholds used : {t_low:.4f} and {t_high:.4f}")
print(f"Number of regions found : {len(np.unique(region_ids[region_ids >= 0]))}")
print("-------------------------------------------------------------------------------")

############################################################################################
## For testing individual steps ############################################################

#vertices, normals, neighbours, mean_edge_lengths = extract_mesh_data(stl_mesh)
#vertices, curvatures, _ = compute_curvatures(stl_mesh)
#plot_curvature_distribution(curvatures, bins=200, filename="results/distribution_discrete_200.png")
#plot_curvature_3d_triangles(stl_mesh)
#plot_curvature_3d_vertices(stl_mesh)
#bin_centers, density = kernel_estimation(curvatures, n_bins=50, h=0.01)
#analyze_distribution(curvatures, bin_centers, density)
#labels, t_low, t_high = thresholding(vertices, curvatures, n_bins=50, h=0.01, window_size=10)

############################################################################################