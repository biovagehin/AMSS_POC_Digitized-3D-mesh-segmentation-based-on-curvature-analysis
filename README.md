# Proof of Concept of the 3D Segmentation Method by S. Gauthier et al.

This repository contains my proof-of-concept work carried out as part of the M2 Advanced Manufacturing & Smart Systems program (ENS Paris-Saclay). It consists of implementing the 3D object segmentation method proposed by Gauthier, S. et al. in their article *"Digitized 3D mesh segmentation based on curvature analysis"*, relying on **local curvature analysis**. 

You will find in this folder:
- `curvature_based_segmentation.py`: the code implementing this method
- `Report_curvature_based_segmentation`: the report for this proof-of-concept project
- `dataset`: this folder contains .stl files to test the algorithm
- `results`: this folder contains the segmentation results on the dataset parts
- `reference`: this folder contains the reference paper as well as other references used for the implementation

To use the code, you can follow these following steps:
- Open `curvature_based_segmentation.py` and adjust the parameters at line 866 (for example: `segmentation(stl_mesh, n_bins=200, h=0.01, window_size=8, min_density=0.02, n_erosions=6)`)
- Run `curvature_based_segmentation.py`
- Enter in the terminal the path to the .stl file you want to test (for example: `dataset\LURPart.stl`)



---

Created by **Biova Géhin** as part of the course *"Techniques and Tools for Proof of Concept"* in the **M2 AMSS** program.


Article reference :
Gauthier, S., Puech, W., Bénière, R., & Subsol, G. (2017). Digitized 3D mesh segmentation based on curvature analysis. Electronic Imaging, 2017(20), 33–38. https://doi.org/10.2352/ISSN.2470-1173.2017.20.3DIPM-005