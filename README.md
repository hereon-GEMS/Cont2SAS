# Description

Software package for calculating Small Angle Scattering (SAS) pattern from continuum simulations.

# Installation

**Run from source**

**Required packages**

# Steps to run
**Mesh generation**

1. define simulation box (only rectangular possible)
2. define node positions
3. define element centres 
4. define connectivity matrix
4. define element type (only lagrangian of order 1 and 2 possible)

- length_a, length_b, length_c = dimension of simulation box (a,b,c -> x,y,z)
- nx, ny, nz = num elements in x,y,z directions
- el_type = type of element (allowed: 'lagrangian')
- el_order=order of element (allowed: 1 or 2)
- update_val= whether to rewrite the structure or not (preferred= 'True')
- plt_node=decision to plot nodes (preffered: 'False') 
plt_cell=decision to plot element centers (preffered: 'False')
plt_mesh=decision to plot mesh (preffered: 'False')



**Assign SLD values to nodes**

**Calculation of SAS patterns**

**Calculation of effective cross-section**

# Model structures

# FEM simulation using Moose
