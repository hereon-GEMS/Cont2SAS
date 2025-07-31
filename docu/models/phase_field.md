# Phase field

This page documents the workflow for creating the input xml related to the `phase_field` model, i.e. model_phase_field.xml. The phase field simulation is done using moose in a docker container. The output is copied to the `moose_read` folder and postprocessed for chosen time steps to input in the `Cont2Sas` software. Postprocessed output is saved in `moose` folder. For each chosen time steps from moose, the `Cont2Sas` software recreates the mesh, assigns the SLD, calculates the SAS pattern.


Relevant input xml: model_phase_field.xml

# Create FEM simulation with moose

## Install docker
```
# install docker
snap install docker
# check installation
docker --version
# check list of groups
groups
```

## Check docker permission

```
# check docker permission
ls -l /var/run/docker.sock
# expected output
# srw-rw---- 1 root docker 0 Jul 30 14:52 /var/run/docker.sock
# if yes jump to xxx
# if not change docker permission
sudo systemctl restart docker
sudo service docker restart
sudo chown root:docker /var/run/docker.sock
ls -l /var/run/docker.sock
```
<!-- 
```
# other commands (taken from history, to be reorganised)
ls -l /var/run/docker.sock
sudo systemctl restart dicker
sudo service docker restart
sudo chown root:docker /var/run/docker.sock
ls -l /var/run/docker.sock
docker ps
docker volume create projects
docker run -it -v projects:/projects idaholab/moose:latest
``` -->

## Create docker container for moose
```
# create volume for data saving
docker volume create projects
# run docker image idaholab/moose:latest
# -it for interactive mode
# -v for attaching volume created by last command
# :/projects creates a directory in the container
docker run -it -v projects:/projects idaholab/moose:latest
# go to projects directory
cd /projects
# copy inputs required for phase field
moose-opt --copy-inputs phase_field
# go to phase_field directory
cd moose/phase_field
# make new directory dor Cont2SAS
mkdir C2S
# exit for docker container
exit
```

## Check docker container name and id
```
# check running dockers
docker ps
```

Example output:

| CONTAINER ID | IMAGE | COMMAND | CREATED | STATUS | PORTS | NAMES |
|----------|----------|----------|----------|----------|----------|----------|
| 4827db3ef5b5  | idaholab/moose:latest  | "/bin/bash -l -c bas…"  | 3 hours ago  | Exited (0)  | 58 seconds ago  | exciting_perlman  |

1. Container name = ${NAMES} (= exciting_perlman in example)
2. Container id = ${CONTAINER ID} (= 4827db3ef5b5 in example)

Note:

The following commands with use container id, which will be refered to as `${cont_id}`. One may also use container name instead.

## Run moose

```
# copy input scripts to docker container
cd $C2S_HOME
docker cp moose_read/FeCr.i ${cont_id}:projects/moose/phase_field/C2S/
# start the container
docker start -ai ${cont_id}
# go to directory
projects/moose/phase_field/C2S/
# run simulation
mpiexec -n 4 --allow-run-as-root moose-opt -i FeCr.i
# exit from container
exit
# copy output to local machine
cd $C2S_HOME
docker cp ${cont_id}:projects/moose/phase_field/C2S/FeCr_out.e moose_read/
```

## Read moose output

```
# go to moose output file location
cd $C2S_HOME/moose_read
# create hdf files calculating sld distribution from moose output
# also rearranges connectivity as per Cont2SAS from moose
python exodus_reader.py
# output for selected time steps are in $C2S_HOME/moose/
# go back to repository home
cd $C2S_HOME
```

## Copy phase_field.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_ball.xml xml/
nano xml/model_ball.xml
```

## Edit phase_field.xml

- name = name of the model (= 'spinodal_fe_cr')
- time = simulation time (should be amongst selected time steps in exodus_reader.py)
- qclean sld = SLD outside simulation box (check in hdf file in moose dir)

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)