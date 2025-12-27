# paper-simulating-particle-motion-2025
This repository contains data and code used to generate results for a manuscript in preparation, developed during a 2025 Research Experience for Undergraduates in the Department of Computational and Applied Mathematics at Rice University, titled “Simulating Particle Motion on Unstructured Meshes Using Inverse Isoparametric Mapping and Bilinear Interpolation.”

You will need to run the code within a Python environment. All scripts were tested using Python 3.12.

## Manuscript Data 
The nodal spatial and temporal data used in this repository were generated using the TrixiShallowWater.jl framework and were provided by Andrew Winters and Patrick Ersing. These data are included in the following files and directories:
- `basic_Trixi_mesh.txt`: Spatial mesh geometry data.
- `Trixi_mesh_100_timesteps/`: Time-resolved velocity field across 100 snapshots.

## Manuscript Codes
The scripts are listed in the order of appearance of their corresponding figures in the manuscript.
- **Figure 1** — `rk2.py`: Demonstrates RK2 timestep sensitivity by comparing numerical trajectories against the exact solution for a simple test ODE. 
- **Figure 5** — `quiver.py`: Generates a quiver visualization of the prescribed velocity field used for the structured-mesh constant-velocity test case. 
- **Figure 6** — `multi.py`: Computes and plots multiple particle trajectories under the same constant velocity field.
- **Figure 7** — `taylor.py`: Simulates particle trajectories in the time-dependent Taylor–Green vortex and compares RK2-advected paths against the analytical ones. Trajectory errors are also reported.
- **Figure 8** — `trixi_single.py`: Computes a single-particle trajectory on the unstructured quadrilateral mesh using the time-averaged velocity field over the provided timesteps. 
- **Figure 9** — `trixi_multi.py`: Produces the snapshot sequence of instantaneous velocity fields and multi-particle trajectories at selected times (t = 0, 5, 15, 30). 
- **Figure 10** — `trixi_oil.py`: Runs the 2000-particle “oil spill” dispersion experiment with vertically stacked circular seeding and advects particles through the time-dependent velocity data. 

