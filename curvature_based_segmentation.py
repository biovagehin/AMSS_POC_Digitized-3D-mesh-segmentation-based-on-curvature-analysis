#### Curvature-Based Segmentation

## STEP 0 : Libraries

import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




## STEP 1 : Load STL File

file_path = input("Enter the path to your STL file: ").strip()
stl_mesh = mesh.Mesh.from_file(file_path) # to load the STL file


# Visualisation of the mesh

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


## Affichage infos structure de stl_mesh
## Nombre de triangles
# print("\nNumber of triangles:")
# print(len(stl_mesh))  # N triangles

# # Premier triangle -> coordonées des 3 sommets
# print("\nFirst triangle vertices (x, y, z):")
# print(stl_mesh.vectors[0])

# # Normale du premier triangle -> coordonnées du vecteur normal
# print("\nNormal of the first triangle:")
# print(stl_mesh.normals[0])







## STEP 2 : Compute Local Curvature at Each Vertex

def local_curvature(vertex, normal, neighbors, neighbor_normals):
    # calcul de la courbure en un sommet, par une régression linéaire sur les courbures par rapport aux sommets voisins
    # k(t_i) = <(P_i - P), (N_i - N)> / ||P_i - P||^2
    # k = k_1 * cos^2(theta) + k_2 * sin^2(theta)
    # theta : angle entre la direction principale de courbure et la direction (P_i - P)
    
    P = vertex
    N = normal

    # calcul des k_i et t_i pour chaque voisin
    k_values = [] # liste des courbures locales k_i
    t_vectors = [] # liste des vecteurs t_i = P_i - P

    for i in range(len(neighbors)):
        P_i = neighbors[i]
        N_i = neighbor_normals[i]
        d_i = P_i - P
        t_i = (d_i - np.dot(d_i, N) * N) / np.linalg.norm(d_i - np.dot(d_i, N) * N) #normalisé

        k_i = (np.dot(d_i, N_i - N)) / (np.linalg.norm(d_i)**2)
        k_values.append(k_i)
        t_vectors.append(t_i)
    
    # Régression linéaire pour trouver k_1 et k_2