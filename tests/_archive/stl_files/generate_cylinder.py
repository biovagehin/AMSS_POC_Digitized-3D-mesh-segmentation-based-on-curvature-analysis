import numpy as np
from stl import mesh

def generate_dense_cylinder_stl(filename="cylinder_dense.stl",
                                radius=10.0, height=20.0,
                                n_circular=100, n_height=50, n_radial=20):
    """
    Génère un cylindre STL dense, maillage uniforme sur la surface et les bases.

    n_circular : segments autour du cylindre
    n_height : segments verticaux sur la paroi
    n_radial : divisions radiales pour la base
    """
    vertices = []
    faces = []

    # --- Paroi ---
    theta = np.linspace(0, 2*np.pi, n_circular, endpoint=False)
    z_vals = np.linspace(0, height, n_height + 1)
    for z in z_vals:
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            vertices.append([x, y, z])
    vertices = np.array(vertices)

    # Triangles paroi
    for i in range(n_height):
        for j in range(n_circular):
            next_j = (j + 1) % n_circular
            lower1 = i * n_circular + j
            lower2 = i * n_circular + next_j
            upper1 = (i+1) * n_circular + j
            upper2 = (i+1) * n_circular + next_j
            faces.append([lower1, lower2, upper2])
            faces.append([lower1, upper2, upper1])

    # --- Base z=0 ---
    base_vertices_start = len(vertices)
    for r in np.linspace(0, radius, n_radial+1)[1:]:
        for t in theta:
            vertices = np.vstack([vertices, [r*np.cos(t), r*np.sin(t), 0]])
    # centre du cercle
    vertices = np.vstack([vertices, [0,0,0]])
    center_idx = len(vertices)-1

    # Triangles base
    for i_r in range(n_radial):
        for i_t in range(n_circular):
            next_t = (i_t + 1) % n_circular
            if i_r == 0:
                idx1 = base_vertices_start + i_t
                faces.append([center_idx, idx1, base_vertices_start + next_t])
            else:
                inner_start = base_vertices_start + (i_r-1)*n_circular
                outer_start = base_vertices_start + i_r*n_circular
                faces.append([inner_start + i_t, inner_start + next_t, outer_start + next_t])
                faces.append([inner_start + i_t, outer_start + next_t, outer_start + i_t])

    # --- Sommet z=height ---
    top_vertices_start = len(vertices)
    for r in np.linspace(0, radius, n_radial+1)[1:]:
        for t in theta:
            vertices = np.vstack([vertices, [r*np.cos(t), r*np.sin(t), height]])
    # centre du sommet
    vertices = np.vstack([vertices, [0,0,height]])
    top_center_idx = len(vertices)-1

    # Triangles sommet
    for i_r in range(n_radial):
        for i_t in range(n_circular):
            next_t = (i_t + 1) % n_circular
            if i_r == 0:
                idx1 = top_vertices_start + i_t
                faces.append([top_center_idx, top_vertices_start + next_t, idx1])
            else:
                inner_start = top_vertices_start + (i_r-1)*n_circular
                outer_start = top_vertices_start + i_r*n_circular
                faces.append([inner_start + i_t, outer_start + next_t, outer_start + i_t])
                faces.append([inner_start + i_t, inner_start + next_t, outer_start + next_t])

    # Création du STL
    cylinder = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cylinder.vectors[i][j] = vertices[f[j]]

    cylinder.save(filename)
    print(f"Cylinder dense saved to {filename} with {len(vertices)} vertices and {len(faces)} triangles.")

# Exemple
generate_dense_cylinder_stl(filename="cylinder_fine.stl", n_circular=80, n_height=15, n_radial=8)
