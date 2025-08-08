# Description of code

The main code is written in two folders:

1. src
2. lib

Generation scripts for generating data and plots are kept in `models` folder.

## lib folder

The lib folder hosts all the necessary functions for the different steps executed by this program, controlled by the scripts in the `src` folder.

Contents:

1. struct_gen.py: Contains functions for the mesh generation step (required by src/struct_gen.py, src/sim_gen.py, src/scatt_cal.py, src/sig_eff.py)
2. simulation.py: Contains functions for the SLD assignment step (required by src/sim_gen.py, src/sig_eff.py)
3. scatt_cal.py: Contains functions for the SAS pattern calculation step (required by src/scatt_cal.py)
4. sig_eff.py: Contains functions for the effective scattering cross section calculation step (required by sig_eff.py)
5. xml_gen.py: Contains functions for generating `xml` files that are then used as input by the `*_gen.py` scripts in the `models` folder (required by all generation scripts in `models` folder)

## src folder

The src folder contains scripts that generate the numerical output data:

1. struct_gen.py: mesh generation scripts
2. sim_gen.py: sld assign script
3. scatt_cal.py: SAS pattern calculation script
4. sig_eff.py: effective cross-section script

## models folder

The models folder contains two scripts for each model that execute the whole [workflow](../general/workflow.md) to generate data and plot the results.
