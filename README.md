![logo](logo/Cont2Sas.png)

# Calculation of SAS pattern from simulated continuum structures

## Description

Software package for calculating Small Angle Scattering (SAS) parameters from continuum simulations.

It can calculate

1. SAS intensity ($I$ vs. $Q$)
2. effective scattering cross-section ($\sigma_\text{eff}$), i.e. the count rate of the scattered radiation per unit flux.

It takes distribution of scattering length density (SLD) as input provided by FEM simulation softwares, e.g., [MOOSE](https://mooseframework.inl.gov/) [Exodus files](https://mooseframework.inl.gov/source/outputs/Exodus.html).

Workflow:

1. Mesh generation
2. SLD assignment
3. SAS pattern calculation
4. effective scattering cross section calculation

For validation, the SLD distribution of known structures can be generated and assigned for which the scatering quantities can be calculated analytically.

As a functional test case, an exemplary MOOSE simulation is also included as `hdf` snapshots out of the Exodus file.

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
To check if thedownload has worked, you can run
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

Remove conda environment, git repository

```
conda remove --name Cont2Sas --all
```

##### Option 2: Using conda and pip

###### Install

```
conda create -n Cont2Sas python=3.12.1
conda activate Cont2Sas
pip install -r requirements.txt
```

###### Uninstall

```
pip uninstall -r requirements.txt
deactivate
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

##### Option 4: Using pip (not recommended)

###### Install

```
pip install -r requirements.txt
```

###### Uninstall

```
pip uninstall -r requirements.txt
```

## Getting started

Some models are provided for generating or simulating nanostructures. From these structures SAS patterns can be calculated. One of the models also facilitates calculation of effective cross-section. 

To run `Cont2SAS` with provided models and parameters, read the [getting started](./getting-started.md) guide.

To check detailed documentation, [click](./docu/index.md) here.

## Acknowledgement

This work was supported by the consortium DAPHNE4NFDI in the context of the work of the
NFDI e.V. The consortium is funded by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) - project number 460248799.

## License

<a href="https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/tree/develop?ref_type=heads"> Cont2Sas </a> 
© 2025 by 
<a href="https://codebase.helmholtz.cloud/arnab.majumdar"> Arnab Majumdar </a> is licensed under 
<a href="https://creativecommons.org/licenses/by/4.0/"> CC BY 4.0 </a>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="Description" width="1" height="1">
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="width: 0.2em;height:0.2em;margin-left: .1em;">
<img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">

See legal code locally: 
<a href="https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/LICENSE?ref_type=heads"> 
LICENSE
</a>
