import numpy as np
from stl import mesh

def create_cube_stl(filename='cube.stl', size=1.0, n_div=10):
    """
    Génère un cube STL avec maillage fin.

    Parameters:
    - filename : nom du fichier STL à sauvegarder
    - size : longueur d'un côté du cube
    - n_div : nombre de subdivisions par arête (maillage plus fin si n_div grand)
    """
    # Crée un tableau pour les triangles
    triangles = []

    # Fonction utilitaire pour créer deux triangles par carré
    def add_quad(p0, p1, p2, p3):
        # deux triangles pour chaque carré
        triangles.append([p0, p1, p2])
        triangles.append([p0, p2, p3])

    # Génération des points sur chaque face
    lin = np.linspace(0, size, n_div+1)

    # Faces XY (z=0 et z=size)
    for z in [0, size]:
        for i in range(n_div):
            for j in range(n_div):
                p0 = [lin[i],   lin[j],   z]
                p1 = [lin[i+1], lin[j],   z]
                p2 = [lin[i+1], lin[j+1], z]
                p3 = [lin[i],   lin[j+1], z]
                add_quad(p0,p1,p2,p3)

    # Faces XZ (y=0 et y=size)
    for y in [0, size]:
        for i in range(n_div):
            for j in range(n_div):
                p0 = [lin[i], y,   lin[j]]
                p1 = [lin[i+1], y, lin[j]]
                p2 = [lin[i+1], y, lin[j+1]]
                p3 = [lin[i], y,   lin[j+1]]
                add_quad(p0,p1,p2,p3)

    # Faces YZ (x=0 et x=size)
    for x in [0, size]:
        for i in range(n_div):
            for j in range(n_div):
                p0 = [x, lin[i],   lin[j]]
                p1 = [x, lin[i+1], lin[j]]
                p2 = [x, lin[i+1], lin[j+1]]
                p3 = [x, lin[i],   lin[j+1]]
                add_quad(p0,p1,p2,p3)

    # Création du mesh
    cube_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        cube_mesh.vectors[i] = np.array(tri)

    # Sauvegarde
    cube_mesh.save(filename)
    print(f"Cube STL généré : {filename}, maillage {n_div}x{n_div} par face")

# Exemple d'utilisation : cube de 1x1x1 avec 20 subdivisions
create_cube_stl(filename='cube.stl', size=20, n_div=30)
