# paper-simulating-particle-motion-2025
This repository contains data and code used to generate results for a manuscript in preparation, developed during a 2025 Research Experience for Undergraduates in the Department of Computational and Applied Mathematics at Rice University, titled “Simulating Particle Motion on Unstructured Meshes Using Inverse Isoparametric Mapping and Bilinear Interpolation.”

You will need to run the code within a Python environment. All scripts were tested using Python 3.12.

## Manuscript Data 
The nodal spatial and temporal data used in this repository were generated using the TrixiShallowWater.jl framework and were provided by Andrew Winters and Patrick Ersing. These data are included in the following files and directories:
- `basic_Trixi_mesh.txt`: Spatial mesh geometry data.
- `Trixi_mesh_100_timesteps/`: Time-resolved velocity field across 100 snapshots.

## Manuscript Codes
The scripts are listed in the order corresponding to their associated figures in the manuscript.
- `rk2.py`:
- `quiver.py`:
- `multi.py`:
- `taylor.py`:
- `trixi_single.py`:
- `trixi_multi.py`:
- `trixi_oil.py`:
