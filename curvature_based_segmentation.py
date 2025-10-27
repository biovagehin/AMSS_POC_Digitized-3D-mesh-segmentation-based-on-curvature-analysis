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



## Affichages infos structure de stl_mesh
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

