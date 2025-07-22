![](logo/Cont2Sas.png)

# Calculation of SAS pattern from simulated continuum structures

## Description

Software package for calculating Small Angle Scattering (SAS) pattern from continuum simulations.

<!-- ![Alt text](logo/Cont2Sas.png) -->


## Installation

### Source code

#### Download the repository

```
git clone git@codebase.helmholtz.cloud:arnab.majumdar/continuum-to-scattering.git
cd continuum-to-scattering
```

### Required packages

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
pip -m venv Cont2Sas
source Cont2Sas\bin\activate
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

#### Other packages

##### Sassena

Download app image and use

Default location: ``Sassena/sassena.AppImage``

```
# download app image 
mkdir -p ./Sassena
wget -O ./Sassena/Sassena.AppImage https://codebase.helmholtz.cloud/api/v4/projects/6801/packages/generic/sassena/v1.9.2-f31e3882/Sassena_CPU.AppImage
# run sassena
chmod +x ./Sassena/Sassena.AppImage
mpirun -n 2 ./Sassena/Sassena.AppImage --help
```

#### Define environment variable

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

## Getting started

Some models are provided for generating or simulating nanostructures. From these structures SAS patterns can be calculated. One of the models also facilitates calculation of effective cross-section. 

To run `Cont2SAS` with provided models and parameters, read the [getting started](./getting-started.md) guide.

## Acknowledgement

This work was supported by the consortium DAPHNE4NFDI in the context of the work of the
NFDI e.V. The consortium is funded by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) - project number 460248799. 

## License

<a href="https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/tree/develop?ref_type=heads">
    Cont2Sas
</a> 
© 2025 by 
<a href="https://codebase.helmholtz.cloud/arnab.majumdar"> Arnab Majumdar 
</a> is licensed under 
<a href="https://creativecommons.org/licenses/by/4.0/"> 
CC BY 4.0 
</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">
<img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">

See legal code locally: 
<a href="https://codebase.helmholtz.cloud/arnab.majumdar/continuum-to-scattering/-/blob/develop/LICENSE?ref_type=heads"> 
LICENSE
</a>