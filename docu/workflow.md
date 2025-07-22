# Steps to run

1. generate mesh
2. assign SLD
3. calculate SAS pattern
4. calculate effective cross-section

## Mesh generation

### Copy input xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/struct.xml xml/
nano xml/struct.xml
```

### Edit input xml

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

### Generate mesh

```
python ./src/struct_gen.py
```

### Output data location

Output is saved in `data` `->` `lengthx_lengthy_lengthz_nx_ny_nz_eltype_elorder` `->` `structure` folder in `struct.h5` file.

#### Run using xml

##### Define struct.xml

1. create struct.xml in xml folder
2. define the following (find [template](https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/xml/Template/struct.xml?ref_type=heads)):

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

- Folder name = ``${proj_home}/data``

### Assign SLD values to nodes

#### Functionalities

1. define time step (only cons time steps possible)
2. read mesh info 
3. assign slds to nodes
4. repeat 3 for all time steps

#### Run using xml

##### Define simulation.xml

1. create simulation.xml in xml folder
2. define the following (find [template](https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/xml/Template/simulation.xml?ref_type=heads)):

    - sim_model = name of the model (e.g.'sld_grow')
    - dt = time step length
    - t_end = end time (start time is 0)
    - n_ensem = number of structures per time step (ensemble of structure in one time step)

##### Define model xml

1. create model_{modelname}.xml in xml folder
2. need to create your own
3. see below for example models

##### Assign sld to nodes

Run following code:

``` 
cd $proj_home

$python ./src/simulation.py
```

##### Output

Check the output in following location:

- Folder name = ``${proj_home}/data/${struct_dir}/simulation/${simu_dir}/${model_dir}``

### Calculation of SAS patterns

#### Functionalities

1. read mesh info
2. read assigned slds 
3. calculate SAS pattern numerically
4. repeat 3 for all time steps

#### Run using xml

##### Define scatt_cal.xml

1. create scatt_cal.xml in xml folder
2. define the following (find [template](https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/xml/Template/scatt_cal.xml?ref_type=heads)):

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

##### Calculate SAS pattern

Run following code:

``` 
cd $proj_home

$python ./src/scatt_cal.py
```

##### Output

Check the output in following location:

- Folder name = ``${proj_home}/data/${struct_dir}/simulation/${simu_dir}/${model_dir}/${time}``

### Calculation of effective cross-section

#### Functionalities

1. generate pixels of detector
2. read SAS pattern and detctor pixel info 
3. calculate effective cross-section
4. repeat 2,3 for all time steps

#### generate pixels of detector

This step is required only of ``detector.h5`` does not exist in ``/detector_geometry/${instru_name}_${facility_name}/``.

``` 
cd $proj_home/detector_geometry/${instru_name}_${facility_name}/

$python ./simu_detector.py

cd $proj_home
```

The output ``detector.h5`` contains following:

- nx, ny = num pixels in detector in x, y dimensions
- dx, dy = pixel width in x and y dimensions
- bs_wx, bs_wy = beam stopper width in x and y dimensions

#### Run using xml

##### Define sig_eff.xml

1. create sig_eff.xml in xml folder
2. define the following (find [template](https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/xml/Template/sig_eff.xml?ref_type=heads)):

    - instrument = instrument name (allowed: 'SANS-1')
    - facility = name of facility (allowed: 'MLZ')
    - distance = distance between detector and sample
    - wl = wavelength of neutron
    - beam_center_coord = vector defining center of beam w.r.t. detector center (e.g. np.array([0, 0, 0]))

##### Calculate effective cross-section

Run following code:

``` 
cd $proj_home

$python ./src/sig_eff.py
```

##### Output

Check the output in following location:

- Folder name = ``${proj_home}/data/${struct_dir}/simulation/${simu_dir}/${model_dir}/${time}``