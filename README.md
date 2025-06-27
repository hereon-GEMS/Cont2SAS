# Calculation of SAS pattern from simulated continuum structures

## Description

Software package for calculating Small Angle Scattering (SAS) pattern from continuum simulations.

## Installation

### Run from source

### Required packages

## Steps to run
### Mesh generation

#### Functionalities

1. define simulation box (only rectangular possible)
2. define node positions
3. define element centres 
4. define connectivity matrix
4. define element type (only lagrangian of order 1 and 2 possible)

#### Run using xml

##### Define following in the xml file:

- lengths = dimension of simulation box in x,y,z
- num_cell = num elements in x,y,z directions
- element
    - type = type of element (allowed: 'lagrangian')
    - order = order of element (allowed: 1 or 2)
- decision
    - update_val = whether to rewrite the structure or not (preferred= 'True')
    - plot
        - node = decision to plot nodes (preffered: 'False') 
        - cell = decision to plot element centers (preffered: 'False')
        - mesh = decision to plot mesh (preffered: 'False')

##### Generate mesh

Run following code:

``` 
cd $proj_home

$python ./src/struct_gen.py
```

##### Output

Check the output in following location:

- Folder name = ${proj_home}/data


### Assign SLD values to nodes

only linear time step possible

- sim_model = name of the model (e.g.'sld_grow')
- dt = time step length
- t_end = end time (start time is 0)
- n_ensem = number of structures per time step (ensemble of structure in one time step)

for models check below

### Calculation of SAS patterns

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

### Calculation of effective cross-section

detector generation

- nx, ny = num pixels in detector in x, y dimensions
- dx, dy = pixel width in x and y dimensions
- bs_wx, bs_wy = beam stopper width in x and y dimensions

params

- instrument = instrument name (allowed: 'SANS-1')
- facility = name of facility (allowed: 'MLZ')
- distance = distance between detector and sample
- wl = wavelength of neutron
- beam_center_coord = vector defining center of beam w.r.t. detector center (e.g. np.array([0, 0, 0]))

## Model structures

### ball

Description: Sphere

- rad = radius of sphere
- sld = SLD of sphere
- qclean_sld = SLD outside simulation box

### box

Description: Parallelepiped

- sld = SLD of parallelepiped
- qclean_sld = SLD outside simulation box

### bib

Description: Sphere at the ceneter of parallepiped

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### bib_ecc

Description: Sphere off the ceneter of parallepiped

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- ecc = vector w.r.t. simulation box center
    - x = x value of vector
    - y = y value of vector
    - z = z value of vector
- qclean_sld = SLD outside simulation box

### gg

Description: Growth of spherical grain

- rad_0 = starting radius of sphere
- rad_end = end radius of sphere
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### fs

Description: Interdiffusion between spherical grain and its environment

- rad = sphere radius
- sig_0 = starting fuzz value (>=0)
- sig_end = end fuzz value (>=0)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### sld_grow

Description: Change of chemical composition of spherical grain

- rad = sphere radius
- sld_in_0 = starting SLD of sphere
- sld_in_end = end SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box



## FEM simulation

### phase_field
Description: Iron - Chromium spinodal decomposition

- name = name of the model (= 'spinodal_fe_cr')
- time = simulation time
- qclean sld = SLD outside simulation box