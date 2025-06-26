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
- plt_cell=decision to plot element centers (preffered: 'False')
- plt_mesh=decision to plot mesh (preffered: 'False')

**Assign SLD values to nodes**

only linear time step possible

- sim_model = name of the model (e.g.'sld_grow')
- dt = time step length
- t_end = end time (start time is 0)
- n_ensem = number of structures per time step (ensemble of structure in one time step)

for models check below

**Calculation of SAS patterns**

- num_cat = num categories for categorization
method_cat = categorization method (allowed : 'direct', 'extend')
- sassena_exe = Sassena executable location ([Check here](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/))
- mpi_procs = number of mpi processes used by Sassena
- num_threads = number of threads used by Sassena
- sig_file = signal file name (preferred: 'signal.h5', must include '.h5' in the end)
- scan_vec = vector along which Q values will be closen (e.g. np.array([1, 0, 0]))
- Q_range = start and end values of Q (e.g. np.array([0., 1.]), must be float)
- num_points = number of Q points
- num_orientation = number of orientations used for numerical orientational averaging

**Calculation of effective cross-section**

# Model structures

# FEM simulation using Moose
