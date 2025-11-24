#### Curvature-Based Segmentation

## STEP 0 : Libraries

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import curve_fit
from math import cos, sin, atan2
from scipy.spatial import KDTree


## STEP 1 : Load STL File and extract vertices and normals
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

# extract vertices and their normals

def extract_vertices_normals(stl_mesh):
    points = stl_mesh.vectors.reshape(-1, 3)
    face_normals = np.repeat(stl_mesh.normals, 3, axis=0)

    vertices = []  # on va chercher les sommets uniques (dans points il y a des doublons car contient les sommets de chaque triangle)
    normals = []   # normales des sommets, calculées à partir des normales des triangles
    index_map = {} # dictionnaire pour mapper les sommets (uniques) à leurs indices

    for i, v in enumerate(points):
        key = tuple(v)
        if key not in index_map:
            index_map[key] = len(vertices)
            vertices.append(points[i])
            normals.append(face_normals[i])
        else:
            idx = index_map[key]
            normals[idx] = normals[idx] + face_normals[i] # normale du sommet = somme des normales des triangles adjacents

    for i in range(len(normals)):
        normals[i] = normals[i] / np.linalg.norm(normals[i]) # normalisation des normales

    return vertices, normals

vertices, normals = extract_vertices_normals(stl_mesh)



## STEP 2 : Compute Local Curvature at Each Vertex
# calcul de la courbure locale en un sommet donné connaissant ses voisins
def local_curvature(vertex, normal, neighbors, neighbor_normals):
    # calcul de la courbure en un sommet
    # k_i(t_i) = <(P_i - P), (N_i - N)> / ||P_i - P||^2
    # approximations des courbures principales k_1 et k_2 par une régression selon : k(phi) = k_1 * cos^2(phi) + k_2 * sin^2(phi)
    # (phi : angle entre la direction principale de courbure et une direction tangentielle)
    
    P = vertex
    N = normal

    # calcul des k_i et t_i pour chaque voisin
    k_list = [] # liste des courbures locales k_i
    t_list = [] # liste des directions tangentielles t_i associées
    phi_list = [] # liste des angles phi associés aux directions t_i

    for i in range(len(neighbors)):
        Pi = neighbors[i]
        Ni = neighbor_normals[i]
        di = Pi - P
        ti = (di - np.dot(di, N) * N) / np.linalg.norm(di - np.dot(di, N) * N) #normalisé
        ki = (np.dot(di, Ni - N)) / (np.linalg.norm(di)**2)

        if len(t_list) == 0:
            phii = 0.0
        else:
            ref = t_list[0]
            phii = atan2(np.dot(ti, np.cross(N, ref)), np.dot(ti, ref))


        k_list.append(ki)
        t_list.append(ti)
        phi_list.append(phii)
        
    
    # Régression linéaire pour trouver k_1 et k_2
    def curvature_model(phi, k1, k2, phi1):
        return k1 * np.cos(phi - phi1)**2 + k2 * np.sin(phi - phi1)**2


    res, _ = curve_fit(curvature_model, phi_list, k_list, p0=[max(k_list), min(k_list), 0])
    k1, k2, phi1 = res

    return k1, k2, phi1

# calcul des courbures principales sur l'ensemble de l'objet
def all_curvatures(vertices, normals, k_neighbors=20):
    
    tree = KDTree(vertices) # KDTree pour chercher les k plus proches voisins
    
    # calcul des courbures principales en chaque sommet
    k1_list = [] 
    k2_list = [] 
    phi1_list = []

    for i in range(len(vertices)):
        P = vertices[i]
        N = normals[i]
        _, idxs = tree.query(P, k=k_neighbors + 1)
        idxs = idxs[1:]  # pour enlever le point lui-même

        neighbors = [vertices[j] for j in idxs]
        neighbor_normals = [normals[j] for j in idxs]

        k1, k2, phi1 = local_curvature(P, N, neighbors, neighbor_normals) # calcul des courbures locales en P

        k1_list.append(k1)
        k2_list.append(k2)
        phi1_list.append(phi1)

    return k1_list, k2_list, phi1_list

k1_list, k2_list, phi1_list = all_curvatures(vertices, normals, k_neighbors=20)



## STEP 3 : Distribution of curvatures
def plot_curvature_distribution(k1_list, bins=50):

    plt.figure(figsize=(10,6))
    plt.hist(k1_list, bins=bins, density=True, color='pink')
    plt.title("Distribution des courbures principales")
    plt.xlabel("k1")
    plt.ylabel("Densité")
    plt.show()

# affichage de la distribution des courbures
plot_curvature_distribution(k1_list)
