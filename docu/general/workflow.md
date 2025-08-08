# Workflow

The workflow of `Cont2SAS` can be divided into four steps:

1. mesh generation
2. SLD assignment
3. SAS pattern calculation
4. effective cross-section calculation

## Mesh generation

The mesh generation step creates a mesh, which includes the coordinates of nodes, cell center, the connectivity matrix, and the element type (defines interpolation function). 

Relevant files:
1. Input xml: struct.xml
2. Script: src/struct_gen.py

Important points:
1. The generated mesh has to match the mesh used for the simulation, when a third-part program (e.g. MOOSE) is used for the simulation.
2. The connectivity matrix has to match the `Cont2SAS` meshing strategy.
3. The current version always generates a regular meshing. 
4. The current version supports Lagrangian elements of 1st and 2nd order.

Workflow:
1. [Copy struct.xml template](#copy-structxml-template)
2. [Edit struct.xml](#edit-structxml)
3. [Generate mesh](#generate-mesh)
4. [Check output](#output-data-location)

### Copy struct.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy from template to xml folder
cp xml/Template/struct.xml xml/
# open with favourite editor (e.g. nano)
nano xml/struct.xml
```

### Edit struct.xml

- lengths = dimension of simulation box in x,y,z (recommended: Angstrom)
- num_cell = number of elements in x,y,z directions
- element
    - type = type of element (allowed: 'lagrangian')
    - order = order of element (allowed: 1 or 2)
- decision
    - update_val = whether to rewrite the structure or not (preferred: 'True')
    - plot
        - node = whether to plot nodes (preferred: 'False') 
        - cell = whether to plot element centers (preferred: 'False')
        - mesh = whether to plot mesh (preferred: 'False')

### Generate mesh

```bash
# run from main folder
cd $C2S_HOME
# run script to generate mesh 
python ./src/struct_gen.py
```

### Output data location

The mesh is saved in `data/${lengthx}_${lengthy}_${lengthz}_${nx}_${ny}_${nz}_${eltype}_${elorder}/structure/struct.h5`.

## SLD assignment

The SLD assignment step assigns scattering length density (SLD) values to the nodes. For generated models, a formula is defined in `lib/simulation.py` for assigning SLD values. For simulated models, the SLD values are read from `hdf5` files, which are created after postprocessing the simulated results.

Relevant files:
1. Input xml: simulation.xml and appropriate model xmls
2. Script: src/sim_gen.py

Important points:
1. For different models, a different model xml input is used (see below for reference).
2. For a new user defined model, a new model xml file must be added.

Workflow:
1. [Copy simulation.xml template](#copy-simulationxml-template)
2. [Edit simulation.xml](#edit-simulationxml)
3. [Prepare model xml](#prepare-model-xml)
4. [Assign sld to nodes](#assign-sld-to-nodes)
5. [Check output](#output-data-location-1)

### Copy simulation.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy from template to xml folder
cp xml/Template/simulation.xml xml/
# open with favourite editor (e.g. nano)
nano xml/simulation.xml
```

### Edit simulation.xml

- sim_model = name of the model
- dt = time step length
- t_end = end time (start time is 0) (recommended unit: seconds)
- n_ensem = number of structures per time step (ensemble of structures in one time step)

### Prepare model xml

The necessary steps differ between the provided simulation models and are given in detail on the following pages:

1. [ball](../models/ball.md)
2. [box](../models/box.md)
3. [bib](../models/bib.md)
4. [bib_ecc](../models/bib_ecc.md)
5. [gg](../models/gg.md)
6. [fs](../models/fs.md)
7. [sld_growth](../models/sld_grow.md)
8. [phase_field](../models/phase_field.md)

After this step, the workflow converges again for all models and is described in the following.

### Assign SLD to nodes

```bash
# run from main folder
cd $C2S_HOME
# run script to assign sld to nodes
python src/sim_gen.py
```

### Output data location

The SLD distribution is saved in `data/${lengthx}_${lengthy}_${lengthz}_${nx}_${ny}_${nz}_${eltype}_${elorder}/simulation/${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}/${model_dir}/${time_dir}/${ensem_dir}/sim.h5`.

## SAS pattern calculation

The SAS pattern calculation step converts the continuous structure to a collection of pseudo atoms with certain scattering lengths and calculates the SAS pattern from these pseudo atoms.

Relevant files:
1. Input xml: scatt_cal.xml
2. Script: src/scatt_cal.py

Important points:
1. This step uses Sassena under the hood

Workflow:
1. [Copy scatt_cal.xml template](#copy-scatt_calxml-template)
2. [Edit scatt_cal.xml](#edit-scatt_calxml)
3. [Calculate SAS pattern](#calculate-sas-pattern)
4. [Check output](#output-data-location-2)

### Copy scatt_cal.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy from template to xml folder
cp xml/Template/scatt_cal.xml xml/
# open with favourite editor (e.g. nano)
nano xml/scatt_cal.xml
```

### Edit scatt_cal.xml

- discretization
    - num_cat = number of categories for categorization
    - method_cat = categorization method (allowed: 'direct', 'extend')
- sassena
    - sassena_exe = [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/) executable location ([check here for install instructions](../../README.md))
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

```bash
# run from main folder
cd $C2S_HOME
# run script to discretize and calculate SAS pattern
python ./src/scatt_cal.py
```

### Output data location

The pseudo atoms with discretized SLD values are saved in `data/${lengthx}_${lengthy}_${lengthz}_${nx}_${ny}_${nz}_${eltype}_${elorder}/simulation/${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}/${model_dir}/${time_dir}/${ensem_dir}/scatt_cal_cat_${method_cat}_${num_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}/scatt_cal.h5`.

The SAS pattern is saved in `data/${lengthx}_${lengthy}_${lengthz}_${nx}_${ny}_${nz}_${eltype}_${elorder}/simulation/${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}/${model_dir}/${time_dir}/Iq_cat_${method_cat}_${num_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}.h5`.

## Effective cross-section calculation

The effective cross-section calculation step calculates the time evolution of the effective cross-section from the SAS patterns at different time steps.

Relevant files:
1. Input xml: sig_eff.xml
2. Script: src/sig_eff.py

Important points:
1. Detector geometry and experimental conditions are required.
2. The only relevant example model contained in this repository is [sld_grow](../models/sld_grow.md).

Workflow:
1. [Generate pixelated detector geometry](#generate-pixelated-detector-geometry)
1. [Copy sig_eff.xml template](#copy-sig_effxml-template)
2. [Edit sig_eff.xml](#edit-sig_effxml)
3. [Calculate effective cross-section](#calculate-effective-cross-section)
4. [Check output](#output-data-location-3)

### Generate pixelated detector geometry

This step is required only if the required `detector.h5` does not exist in `/detector_geometry/${instru_name}_${facility_name}/`.

```bash
# go to detector directory
cd $C2S_HOME/detector_geometry/${instru_name}_${facility_name}/
# generate pixelated detector geometry
python ./simu_detector.py
# change dir to repositoty home
cd $C2S_HOME
```

The output ``detector.h5`` contains following:

- nx, ny = number of pixels in detector in x, y dimensions
- dx, dy = pixel width in x and y dimensions
- bs_wx, bs_wy = beam stop width in x and y dimensions

### Copy sig_eff.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy tempate to xml folder
cp xml/Template/sig_eff.xml xml/
# open xml with favourite editor (e.g. nano)
nano xml/sig_eff.xml
```

### Edit sig_eff.xml

- instrument = instrument name (currently existing: 'SANS-1')
- facility = name of facility (currently existing: 'MLZ')
- d = distance between detector and sample
- lambda = wavelength of the incident radiation
- beam_center = vector from the defining the center position of the direct beam relative to the detector center (e.g. np.array([0, 0, 0]))
    - x component
    - y component
    - z component

### Calculate effective cross-section

```bash
# change to main folder
cd $C2S_HOME
# run script to calculate effective cross-section
python ./src/sig_eff.py
```

### Output data location

The effective cross-section is saved in `data/${lengthx}_${lengthy}_${lengthz}_${nx}_${ny}_${nz}_${eltype}_${elorder}/simulation/${sim_model}_t_end_${t_end}_dt_${dt}_ensem_${ensem}/${model_dir}/sig_eff_cat_${method_cat}_${num_cat}_Q_${Q_start}_${Q_end}_orien__${num_orientation}.h5`.
