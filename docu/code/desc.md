# Description of code

The main code is written in two folders:

1. src
2. lib

Generation scripts for generating data and plots are kept in `models` folder

## src folder

The src folder hosts all scripts that are used to generate data.

Provided scripts are:

1. struct_gen.py: mesh generation scripts
2. sim_gen.py: sld assign script
3. scatt_cal.py: SAS pattern calculation script
4. sig_eff.py: effective cross-section script

## lib folder

The lib folder hosts all scripts that hosts necessary functions.

Provided scripts are:

1. struct_gen.py: required by src/struct_gen.py, src/sim_gen.py, src/scatt_cal.py, src/sig_eff.py
2. simulation.py: required by src/sim_gen.py, src/sig_eff.py
3. scatt_cal.py: required by src/scatt_cal.py
4. sig_eff.py
5. xml_gen.py: required by generation scripts in `models` folder