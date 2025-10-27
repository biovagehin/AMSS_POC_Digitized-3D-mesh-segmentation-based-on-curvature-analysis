# Proof of Concept of the 3D Segmentation Method by S. Gauthier et al.

This repository contains my proof-of-concept work carried out as part of the M2 Advanced Manufacturing & Smart Systems program (ENS Paris-Saclay). It consists of implementing the 3D object segmentation method proposed by Gauthier, S. et al. in their article *"Digitized 3D mesh segmentation based on curvature analysis"*.

This method enables the segmentation of 3D objects using **local curvature analysis**. The main steps of the code *(work in progress)* are as follows:

1. **Loading the STL file**  
   Reading the 3D model to be segmented.

2. **Local curvature computation**  
   Calculating the curvature at each point of the mesh.

3. **Curvature distribution plotting and analysis**  
   Analyzing the distribution of curvature values to identify key areas for segmentation (peaks and valleys detection).

4. **Segmentation**  
   - Edge detection based on curvature variations  
   - Region growing to group homogeneous regions  
   - Post-processing to clean and refine the segmentation

5. **Recursivity**  
   Applying the segmentation multiple times for complex shapes: after the initial segmentation, the code can be applied recursively to sub-meshes to improve the results.



(The repository also includes a folder [`exercice_depot_git`](./exercice_depot_git), which contains a **notebook** demonstrating how to compute a **least squares plane** from a point cloud.  
This served as a preliminary exercise to familiarize myself with creating and managing a Git repository before working on the main project.)



---

Created by **Biova Géhin** as part of the course *"Techniques and Tools for Proof of Concept"* in the **M2 AMSS** program.


Article reference :
Gauthier, S., Puech, W., Bénière, R., & Subsol, G. (2017). Digitized 3D mesh segmentation based on curvature analysis. Electronic Imaging, 2017(20), 33–38. https://doi.org/10.2352/ISSN.2470-1173.2017.20.3DIPM-005