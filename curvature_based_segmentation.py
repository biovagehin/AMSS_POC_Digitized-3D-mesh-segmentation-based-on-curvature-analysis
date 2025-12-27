#### Curvature-Based Segmentation

## STEP 0 : Libraries

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from math import cos, sin, atan2
from scipy.spatial import KDTree


## STEP 1 : Load STL File, Extract Vertices, their Normals, their 1-Ring Neighbours and the Mean Edge Length
file_path = input("Enter the path to your STL file: ").strip()
stl_mesh = mesh.Mesh.from_file(file_path)

# Visualisation of the mesh
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

# extract vertices, their normals, their 1-ring neighbours, and the mean edge length

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
        normals[i] = normals[i] / np.linalg.norm(normals[i]) # normalisation des normales

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

## STEP 2 : Compute Local Curvature at Each Vertex of the Mesh

# compute local curvature at a vertex P given its normal N and its 1-ring neighbours
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


    res, _ = curve_fit(curvature_model, phii_list, ki_list, p0=[max(ki_list), min(ki_list), 0])
    k_max, k_min, phi_max = res

    return k_max

# compute curvatures for all vertices
def all_curvatures(stl_mesh):
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




## STEP 3 : Distribution and visualization of curvatures

# distribution of normalised curvatures (k_max * mean_edge_length)
def plot_curvature_distribution(curvatures, bins=50):
    plt.figure(figsize=(10,6))
    plt.hist(curvatures, bins=bins, color='pink', edgecolor='black')
    plt.title("Distribution des courbures principales maximales normalisées")
    plt.xlabel(f"Courbure principale* Longueur d'arête moyenne")
    plt.ylabel("Nombre d'occurences")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# visualisation of curvatures on the 3D mesh
# triangles are colored according to the mean curvature of their vertices

def plot_curvature_3d(stl_mesh):
    vertices, curvatures, index_map = all_curvatures(stl_mesh)
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
    cbar.set_label("Courbure principale normalisée (sans unité)")

    plt.show()


### RUN SECTION ############################################################################
#vertices, normals, neighbours, mean_edge_lengths = extract_mesh_data(stl_mesh)
#_, curvatures = all_curvatures(stl_mesh)
#plot_curvature_distribution(curvatures)
#plot_curvature_3d(stl_mesh)