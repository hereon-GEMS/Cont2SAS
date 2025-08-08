# Phase field

This page documents the workflow for creating the input xml related to the `phase_field` model, i.e. model_phase_field.xml. The phase field simulation is done using MOOSE in a docker container. The output is copied to the `moose_read` folder and postprocessed for chosen time steps to input in the `Cont2Sas` software. Postprocessed output is saved in `moose` folder. For each chosen time steps from MOOSE, the `Cont2Sas` software recreates the mesh, assigns the SLD, calculates the SAS pattern.

Relevant input xml: model_phase_field.xml

## Create FEM simulation with MOOSE

Use these instructions to create a FEM simulation with MOOSE. The input files are provided in the moose_read folder. The output will be generated in the moose folder.

A group of sample `hdf` outputs are already included in the moose folder of this repository. Skip this step and [jump here](#calculate-the-sas-parameters-with-cont2sas) to calculate the SAS patterns from the provided sample.

### Install docker

```bash
# install docker
snap install docker
# check installation, maybe need to follow [the steps described here](https://docs.docker.com/engine/install/linux-postinstall/)
docker --version
```

### Check docker permission

```bash
# check docker permission
ls -l /var/run/docker.sock
# expected output
# srw-rw---- 1 root docker 0 Jul 30 14:52 /var/run/docker.sock
# if yes jump to next step
# if not change docker permission
sudo systemctl restart docker
sudo service docker restart
sudo chown root:docker /var/run/docker.sock
ls -l /var/run/docker.sock
```

### Create docker container for MOOSE

```bash
# create volume for data saving
docker volume create projects
# run docker image idaholab/moose:latest
# -it for interactive mode
# -v for attaching volume created by last command
# :/projects creates a directory in the container
docker run -it -v projects:/projects idaholab/moose:latest
# go to projects directory
cd /projects
# copy inputs required for phase field from the moose examples
moose-opt --copy-inputs phase_field
# go to phase_field directory
cd moose/phase_field
# make new directory dor Cont2SAS
mkdir C2S
# exit for docker container
exit
```

### Check docker container name and id

```bash
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

The following commands will use container id, which will be referred to as `${cont_id}`. One may also use container name instead.

### Run MOOSE

```bash
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

### Read MOOSE output

```bash
# go to MOOSE output file location
cd $C2S_HOME/moose_read
# create hdf files calculating sld distribution from MOOSE output
# also rearranges connectivity as per Cont2SAS from MOOSE
python exodus_reader.py
# output for selected time steps are in $C2S_HOME/moose/
# go back to repository home
cd $C2S_HOME
```

## Calculate the SAS parameters with Cont2SAS

### Copy phase_field.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_phase_field.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_phase_field.xml
```

### Edit phase_field.xml

- name = name of the model (= 'spinodal_fe_cr') for naming the `hdf` files in the moose directory
- time = simulation time step (should be amongst selected time steps in exodus_reader.py)
- qclean_sld = SLD outside of the simulation box (the value used for the provided model can be found in the hdf files in the moose directory)

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)
