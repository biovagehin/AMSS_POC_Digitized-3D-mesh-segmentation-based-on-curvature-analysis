from solid import *
from solid.utils import *
import math

# Paramètres
cube_size = 10
radius = 1
mesh_quality = 0.5  # plus petit = plus dense

# Création du cube avec congés
def cube_with_fillets(size, fillet_radius, segments=32):
    # Création d'un cube de base
    base = cube(size, center=True)
    
    # On crée des cylindres pour les arrondis aux arêtes
    edges = []
    for x in [-size/2 + fillet_radius, size/2 - fillet_radius]:
        for y in [-size/2 + fillet_radius, size/2 - fillet_radius]:
            for z in [-size/2 + fillet_radius, size/2 - fillet_radius]:
                edges.append(translate([x,y,z])(sphere(r=fillet_radius, segments=segments)))
    
    # Union cube + congés
    cube_filleted = base + sum(edges)
    
    return cube_filleted

# Génération
cube_obj = cube_with_fillets(cube_size, radius, segments=32)

# Export en STL
scad_render_to_file(cube_obj, "cube_filleted.scad", file_header='$fn = 64;')
# Ensuite, convertit le SCAD en STL via OpenSCAD CLI
# openSCAD command: openscad -o cube_filleted.stl cube_filleted.scad
