![logo](logo/Cont2Sas.png)

# Calculation of SAS parameters from simulated continuum structures

## Description

Software package for calculating Small Angle Scattering (SAS) parameters from continuum simulations.

It can calculate

1. SAS intensity ($I$ vs. $Q$)
2. effective scattering cross-section ($\sigma_\text{eff}$), i.e. the count rate of the scattered radiation per unit flux.

It takes a distribution of scattering length density (SLD) values as input that can be provided by FEM simulation softwares, e.g., [Exodus files](https://mooseframework.inl.gov/source/outputs/Exodus.html) provided by [MOOSE](https://mooseframework.inl.gov/).

Workflow:

1. Mesh generation
2. SLD assignment
3. SAS pattern calculation
4. effective scattering cross section calculation

For validation, the SLD distribution of known structures can be generated and assigned, for which the scatering quantities can be calculated analytically.

As a functional test case, an exemplary MOOSE simulation is also included as `hdf` snapshots out of the Exodus file.

Files in repository:

1. `database_sassena`: 
    - Description: Copy of [sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/) database.
    - Purpose: Some files are copied for creating database while calculating from continuum simulated structures.
2. `detector_geometry`
    - Description: Simulated version of detector of instruments (e.g. [SANS-1/MLZ](https://mlz-garching.de/sans-1)).
    - Purpose: Required for calculating effective cross-section $(\sigma_{\text{eff}})$.
3. `docu`
    - Description: Documentation.
    - Purpose: Deatiled description of software.
4. `joss_submission`
    - Description: [JOSS](https://joss.readthedocs.io/en/latest/submitting.html) publication related files.
    - Purpose: Required for publishing in [JOSS](https://joss.readthedocs.io/en/latest/submitting.html).
5. `lib`
    - Description: List of functions required by run scripts in `src` and `models`.
    - Purpose: Required for relevant run scripts in `src` and test scripts in `models`.
6. `logo`
    - Description: logo of `Cont2SAS`.
7. `models`
    - Description: data generation and plot generation scripts for provided models (see [quick start guide](./getting-started.md))
    - Purpose: Required for testing the functionality of `Cont2SAS`.
8. `moose`
    - Description: Output of moose formated as input of `Cont2SAS`
    - Purpose: Required for assigning SLD in mesh for `phase_field` model, reads SLD values from `h5` file for chosen time steps.
9. `moose_read`
    - Description: [MOOSE](https://mooseframework.inl.gov/) input script and python scripts for converting [MOOSE](https://mooseframework.inl.gov/) output to `Cont2SAS` input.
    - Purpose: FEM simulation using [MOOSE](https://mooseframework.inl.gov/) can be performed using docker container and the input scripts and output [exodus files](https://mooseframework.inl.gov/source/outputs/Exodus.html) can be converted to input of `Cont2SAS` using `exodus_reader.py` (see [instructions](./docu/models/phase_field.md))
10. `shell_scripts`
    - Description: All shell scripts
    - Purpose: Install and uninstall
11. `src`
    - Description: Run scripts of `Cont2SAS`
    - Purpose: reads xmls from `xml` folder as input; generates mesh (`struct_gen.py`), assigns SLD (`sim_gen.py`), calculates SAS patterns (`scatt_cal.py`), and effective cross-section (`sig_eff.py`); also used test scripts in `models`.
12. `xml`
    - Description: input xml files.
    - Purpose: Required for specifying input conditions read by run scripts.
13. `.gitignore`
    - Description: Specify files ignored by git 
14. `.pylintrc`
    - Description: Configuration file for pylint
15. `CITATION.cff`
    - Description: Citation metadata
16. `README.md`
    - Description: This file
17. `environment.yml`
    - Description: Packages and their versions installed by conda
18. `getting-started.md`
    - Description: Quick start guide
19. `requirements.txt`
    - Description: Packages and their versions installed by pip

## Installation

### Source code

#### Download the repository

```
git clone git@codebase.helmholtz.cloud:arnab.majumdar/continuum-to-scattering.git
cd continuum-to-scattering
```

#### Define environment variable

To ensure that the provided examples run on the user's computer regardless of the starting directory, the following environment variable is used.

##### Install

```
# define for current session
export C2S_HOME=$PWD
# define for further session
chmod +x ./shell_scripts/*
./shell_scripts/install.sh
```

##### Uninstall

```
# remove env var
./shell_scripts/uninstall.sh
```

### Required packages

#### Sassena

Download app image and use

Default location: ``Sassena/sassena.AppImage``

```
# download app image
mkdir -p ./Sassena
wget -O ./Sassena/Sassena.AppImage https://codebase.helmholtz.cloud/api/v4/projects/6801/packages/generic/sassena/v1.9.2-f31e3882/Sassena_CPU.AppImage
# run sassena
chmod +x ./Sassena/Sassena.AppImage
```
To check if the download has worked, you can run
```
./Sassena/Sassena.AppImage --help
```
which should print out messages about command line options of [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena) and end with an error message.

#### Python packages

##### Option 1: Using conda

###### Install

```
conda env create -f environment.yml
conda activate Cont2Sas
```
###### Uninstall

Remove the conda environment

```
conda deactivate
conda remove --name Cont2Sas --all
```
and delete the source code directory by hand.

##### Option 2: Using conda and pip

###### Install

```
conda create -n Cont2Sas python=3.8.10
conda activate Cont2Sas
pip install -r requirements.txt
```

###### Uninstall

```
pip uninstall -r requirements.txt
conda deactivate
conda remove --name Cont2Sas --all
```

##### Option 3: Using pip and venv

###### Install

```
sudo apt update
sudo apt install python3-venv
python3 -m venv Cont2Sas
source Cont2Sas/bin/activate
pip install -r requirements.txt
```

###### Uninstall

```
pip uninstall -r requirements.txt
deactivate
rm -rf Cont2Sas
```

<!-- ##### Option 4: Using pip (not recommended)

###### Install

```
pip install -r requirements.txt
```

###### Uninstall

```
pip uninstall -r requirements.txt
``` -->

## Getting started

Some models are provided for generating or simulating nanostructures. From these structures SAS patterns can be calculated. One of the models also facilitates calculation of effective cross-section. 

To run `Cont2SAS` with provided models and parameters, read the [getting started](./getting-started.md) guide.

To check detailed documentation, [click](./docu/index.md) here.

## Acknowledgement

This work was supported by the consortium DAPHNE4NFDI in the context of the work of the
NFDI e.V. The consortium is funded by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) - project number 460248799.

## License

One line to give the program's name and an idea of what it does.
Copyright (C) 2025  Arnab Majumdar

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see
<https://www.gnu.org/licenses/>.

See legal code:</br>
<a href="https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/LICENSE?ref_type=heads"> 
LICENSE
</a>

Copyright disclaimer from Helmholtz-Zentrum, Hereon (DE)

Urheberrecht:
Software unterliegt dem Urheberrecht. § 69a Urhebergesetz (UrhG) schützt Software in jeder
Gestaltung, ein- schließlich Entwurfsmaterial, in allen Ausdrucksformen (QC, C, EXE, Module).
Nicht geschützt sind dagegen Ideen und Grundsätze, die dem Werk zugrunde liegen.
Das Urheberpersönlichkeitsrecht ist nach deutschem Recht zwar nicht übertragbar, zulässig
sind jedoch die Einräumung von umfänglichen Nutzungsrechten sowie Vereinbarungen zu
Verwertungsrechten. Legt man z. B. amerikanisches Recht zugrunde, dann ist auch eine Abtretung des gesamten Urheberrechts möglich.