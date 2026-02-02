![logo](logo/logo/Cont2SAS-logo-transparent.png)

# Cont2SAS

## Description

### Software package

Software package for calculating Small Angle Scattering (SAS) parameters from continuum simulations.

It can calculate

1. SAS intensity ($I$ vs. $Q$) and
2. Effective scattering cross-section ($\sigma_\text{eff}$), i.e. the count rate of the scattered radiation per unit flux.

It takes a distribution of scattering length density (SLD) values as input that can be provided by FEM simulation softwares, e.g., [Exodus files](https://mooseframework.inl.gov/source/outputs/Exodus.html) provided by [MOOSE](https://mooseframework.inl.gov/).

Workflow:

1. Mesh generation
2. Scattering length density (SLD) assignment
3. SAS pattern calculation
4. Effective scattering cross-section calculation

The output data are stored in the `data` folder. Its detailed structure is described in the [documentation](./docu/index.md).

For validation, the SLD distribution of known structures can be generated and assigned, from which the scatering quantities can be calculated using this software and compared with analytical expressions. Several models are provided as validation test cases that follow this workflow. 

As a functional test case, snapshots of an Exodus file obtained from exemplary MOOSE simulation are also included as `hdf` files. The SLD values are assigned from these `hdf` snapshot files to a generated mesh and SAS quantities are calculated using this software.

The following test cases are included in this repository:

- Validation test case: Generated model SLD distribution
    - Static models (structures) containing one time step
        - Sphere
        - Cube
        - Sphere at the center of cube
        - Sphere off the center of cube
    - Evolving models (phenomena) containing several time steps
        - Growth of a sphere
        - Interdiffusion of a sphere with its environment
        - Change of the chemical composition of a sphere
- Functional test case: Simulated model SLD distribution
    - Phase field modeling using MOOSE
        - Spinodal decomposition of Fe-Cr

Scripts are provided in the `models` folder that generate data and figures related to different test cases. All cases include a data generation script that stores generated data in `data` folder and a plot script that creates figures in `figure` folder. For the sld_grow model, an additional plot script is provided that creates figure for [JOSS](https://joss.readthedocs.io/en/latest/) publication.

### Files in repository

A description and purpose of the provided files are listed below:

1. `database_sassena`: A copy of the [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/) database of scattering lengths. Some of these files are copied for creating a custom database while calculating the scattering patterns from continuum simulated structures.
2. `detector_geometry`: A representation of the detector of small-angle scattering instruments (e.g. [SANS-1@MLZ](https://mlz-garching.de/sans-1)). This is required for calculating the effective cross-section $(\sigma_{\text{eff}})$.
3. `docu`: Detailed description of this software.
4. `joss_submission`: Files for the publication of this software in the [Journal of Open Source Software (JOSS)](https://joss.readthedocs.io/en/latest/submitting.html).
5. `lib`: Functions required by the run scripts in `src` and `models`. See [documentation](./docu/code/desc.md).
6. `logo`: Logo of `Cont2SAS`.
7. `models`: Data generation and plot generation scripts for the models provided with this software for testing (see [quick start guide](./getting-started.md)).
8. `moose`: Output of a [MOOSE](https://mooseframework.inl.gov/) simulation, reformatted as input for `Cont2SAS`. Required for assigning SLD values in the mesh of the `phase_field` model. Reads the SLD values from `hdf5` files for chosen time steps.
9. `moose_read`: (a) Input script for a [MOOSE](https://mooseframework.inl.gov/) simulation -- instructions how to run the corresponding FEM simulation using a docker container are [included in the detailed documentation](./docu/models/phase_field.md). (b) python scripts for converting the output [exodus files](https://mooseframework.inl.gov/source/outputs/Exodus.html) of the MOOSE simulation into `Cont2SAS` input.
10. `shell_scripts`: Shell scripts that add/remove an environment variable to the user's shell config file for easy execution of the provided examples.
11. `src`: Scripts that read xml files from the `xml` folder as input; generate mesh (`struct_gen.py`), assign SLD (`sim_gen.py`), calculate SAS patterns (`scatt_cal.py`), and effective cross-section (`sig_eff.py`); also used by the `*_gen.py` scripts in `models` (see [documentation](./docu/code/desc.md)).
12. `tests`: Test scripts for one static validation model, one evolving validation model, and one functional simulated model. Checks whether the data get generated, plots get generated, plots are correct, cleans generated files after execution of tests.
13. `xml`: Input xml files that specify variables read by the run scripts.

## Installation

### Source code

#### Download the repository

```bash
git clone git@github.com:hereon-GEMS/Cont2SAS.git
cd Cont2SAS
```

#### Define environment variable

To ensure that the provided examples run on the user's computer regardless of the starting directory, the following environment variable is used.

```bash
# define environment variable for the current session
export C2S_HOME=$PWD
```

##### Install

```bash
# define environment variable for further sessions in the shell config file
chmod +x ./shell_scripts/*
./shell_scripts/install.sh
```

##### Uninstall

```bash
# remove environment variable definition from the shell config file
./shell_scripts/uninstall.sh
```

### Required packages

#### Sassena

Download app image and use

Default location: `Sassena/sassena.AppImage`

```bash
# download app image
mkdir -p ./Sassena
wget -O ./Sassena/Sassena.AppImage https://codebase.helmholtz.cloud/api/v4/projects/6801/packages/generic/sassena/v1.9.2-f31e3882/Sassena_CPU.AppImage
# run sassena
chmod +x ./Sassena/Sassena.AppImage
```
To check if the download has worked, you can run
```bash
./Sassena/Sassena.AppImage --help
```
which should print out messages about command line options of [Sassena](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena) and end with an error message.

#### Python packages

##### Option 1: Using conda

###### Install

```bash
conda env create -f environment.yml
conda activate Cont2Sas
```
###### Uninstall

Remove the conda environment:

```bash
conda deactivate
conda remove --name Cont2Sas --all
```
and delete the source code directory by hand.

##### Option 2: Using conda and pip

###### Install

```bash
conda create -n Cont2Sas -c conda-forge python=3.12.1 mdtraj=1.9.9
conda activate Cont2Sas
# install everything in requirements.txt except mdtraj
grep -v mdtraj requirements.txt > requirements_no_mdtraj.txt && pip install -r requirements_no_mdtraj.txt && rm -r requirements_no_mdtraj.txt
```

###### Uninstall

```bash
grep -v mdtraj requirements.txt > requirements_no_mdtraj.txt && pip uninstall -r requirements_no_mdtraj.txt && rm -r requirements_no_mdtraj.txt
conda deactivate
conda remove --name Cont2Sas --all
```

##### Option 3: Using pip and venv

###### Install

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv Cont2Sas
source Cont2Sas/bin/activate
pip install -r requirements.txt
```

###### Uninstall

```bash
pip uninstall -r requirements.txt
deactivate
rm -rf Cont2Sas
```

### Test installation

To check whether the installation has worked, execute the tests:

```bash
pytest -v
```

## Getting started

Some models are provided for generating or simulating nanostructures. From these structures SAS patterns can be calculated. One of the models also facilitates calculation of effective cross-section. 

To run `Cont2SAS` with the provided models and parameters, read the [getting started](./getting-started.md) guide.

A [detailed documentation](./docu/index.md) is provided in the `docu` folder.

Potential developers, who want to contribute to the project, are advised to refer the [contributor's guide](./CONTRIBUTING.md).

## Referencing Cont2SAS

For the citation information of `Cont2SAS`, please see the [CITATION.cff](./CITATION.cff) file.

## License

The `Cont2SAS` source code is released under the [GNU General Public License Version 3 or later](./LICENSE).

## Acknowledgement

`Cont2SAS` is developed at [Helmholtz-Zentrum Hereon](https://www.hereon.de/) and the development is supported by the consortium [DAPHNE4NFDI](https://www.daphne4nfdi.de/) in the context of the work of the NFDI e.V. The consortium is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248799.
