<!-- <p align="center">
<img src="logo/Cont2Sas.png" alt="Description" width="300"/>
</p> -->

![](logo/Cont2Sas.png)

# Calculation of SAS pattern from simulated continuum structures

<!-- <p align="center">
  <img src="path/to/image.png" alt="Description" width="300"/>
</p> -->

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