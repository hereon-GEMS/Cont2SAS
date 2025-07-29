# Workflow

The workflow of `Cont2SAS` can be divided into four steps:

1. mesh generation
2. SLD assignment
3. SAS pattern calculation
4. Effective cross-section calculation

## Mesh generation

The mesh generation step creates a mesh, which includes the coordinates of nodes, cell center, connectivity matrix, element type. 

Relevant files:
1. Relevant input xml: struct.xml
2. Relevant script: src/struct_gen.py

Important points:
1. Generated mesh should match the mesh used for simulation, when outside software is used (e.g. moose)
2. The connectivity matrix should match the `Cont2SAS` meshing strategy.
3. The current version can only generate regular meshing. 
4. The current version supports only lagrangian elements of 1st and 2nd order.

### Copy struct.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/struct.xml xml/
nano xml/struct.xml
```

### Edit struct.xml

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

## SLD assignment

The SLD assignment step assigns SLD values to the nodes. For generated models, a formula is defined for assigning SLD values. For simulated models, the SLD values are read from `hdf5` files, which is created after postprocessing of simulated results.

1. Relevant input xml: simulation.xml and appropiate model xmls
2. Relevant script: src/sim_gen.py

Important points:
1. For different models, different model xml is input (see below for reference).
2. For a new user defined model, definition of model xml must be added.

### Copy simulation.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/simulation.xml xml/
# open with favourite editor (e.g. nano)
nano xml/simulation.xml
```

### Edit simulation.xml

- sim_model = name of the model
- dt = time step length
- t_end = end time (start time is 0)
- n_ensem = number of structures per time step (ensemble of structure in one time step)

### Prepare model xml

Check necessary steps for provided simulation models (sim_model):

1. [ball](../models/ball.md)
2. [box](../models/box.md)
3. [bib](../models/bib.md)
4. [bib_ecc](../models/bib_ecc.md)
5. [gg](../models/gg.md)
6. [fs](../models/fs.md)
7. [sld_growth](../models/sld_grow.md)
8. [phase_field](../models/phase_field.md)

### Assign sld to nodes

Run following code:

```
cd $C2S_HOME
python src/sim_gen.py
```

### Output data location

Output is saved in `data` `->` `lengthx_lengthy_lengthz_nx_ny_nz_eltype_elorder` `->` `simulation` `->` `${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}` `->` `${model_dir}` `->` `${time_dir}` `->` `${ensem_dir}` folder in `sim.h5`

## SAS pattern calculation

### Copy scatt_cal.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/scatt_cal.xml xml/
nano xml/scatt_cal.xml
```

### Edit scatt_cal.xml

- discretization
    - num_cat = num categories for categorization
    - method_cat = categorization method (allowed : 'direct', 'extend')
- sassena
    - sassena_exe = [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/) executable location ([Check here for install instructions](../../README.md))
    - mpi_procs = number of mpi processes used by Sassena
    - num_threads = number of threads used by Sassena
- scatt_cal
    - sig_file = signal file name (preferred: 'signal.h5', must include '.h5' in the end)
    - scan_vec = vector along which Q values will be closen (e.g. np.array([1, 0, 0]))
        - x = x component
        - y = y component
        - z = z component
    - Q_start = start of Q range (must be float)
    - Q_end = end of Q range (must be float)
    - num_points = number of Q points
    - num_orientation = number of orientations used for numerical orientational averaging

### Calculate SAS pattern

```
cd $C2S_HOME
python ./src/scatt_cal.py
```

### Output data location

Output of discretization is saved in `data` `->` `lengthx_lengthy_lengthz_nx_ny_nz_eltype_elorder` `->` `simulation` `->` `${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}` `->` `${model_dir}` `->` `${time_dir}` `->` `${ensem_dir}` `->` `scatt_cal_cat_${method_cat}_${mum_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}` folder in `scatt_cal.h5`

SAS pattern is saved in `data` `->` `lengthx_lengthy_lengthz_nx_ny_nz_eltype_elorder` `->` `simulation` `->` `${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}` `->` `${model_dir}` `->` `${time_dir}` folder in `Iq_cat_${method_cat}_${mum_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}.h5`

## Effective cross-section calculation

### Generate pixelated detector geometry

This step is required only if ``detector.h5`` does not exist in ``/detector_geometry/${instru_name}_${facility_name}/``.

``` 
cd $C2S_HOME/detector_geometry/${instru_name}_${facility_name}/

python ./simu_detector.py

cd $C2S_HOME
```

The output ``detector.h5`` contains following:

- nx, ny = num pixels in detector in x, y dimensions
- dx, dy = pixel width in x and y dimensions
- bs_wx, bs_wy = beam stopper width in x and y dimensions

### Copy sig_eff.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/sig_eff.xml xml/
nano xml/sig_eff.xml
```

### Edit sig_eff.xml

- instrument = instrument name (allowed: 'SANS-1')
- facility = name of facility (allowed: 'MLZ')
- d = distance between detector and sample
- lambda = wavelength of neutron
- beam_center = vector defining center of beam w.r.t. detector center (e.g. np.array([0, 0, 0]))
    - x component
    - y component
    - z component

##### Calculate effective cross-section

Run following code:

``` 
cd $C2S_HOME
python ./src/sig_eff.py
```

##### Output

Effective cross-section is saved in `data` `->` `lengthx_lengthy_lengthz_nx_ny_nz_eltype_elorder` `->` `simulation` `->` `${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}` `->` `${model_dir}` folder in `sig_eff_cat_${method_cat}_${mum_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}.h5`