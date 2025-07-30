# Phase field

This page documents the workflow for creating the input xml related to the `phase_field` model, i.e. model_phase_field.xml. The phase field simulation is done using moose in a docker container. The output is copied to the `moose_read` folder and postprocessed for chosen time steps to input in the `Cont2Sas` software. Postprocessed output is saved in `moose` folder. For each chosen time steps from moose, the `Cont2Sas` software recreates the mesh, assigns the SLD, calculates the SAS pattern.


Relevant input xml: model_phase_field.xml

# Create FEM simulation with moose

Create docker container for moose

```
# install docker
snap install docker
# check installation
docker --version
# check list of groups
groups
```

Check docker permission before running

```
# check docker permission
ls -l /var/run/docker.sock
# expected output
# srw-rw---- 1 root docker 0 Jul 30 14:52 /var/run/docker.sock
# if yes jump to check running dockers
# if not change docker permission
sudo systemctl restart docker
sudo service docker restart
sudo chown root:docker /var/run/docker.sock
ls -l /var/run/docker.sock
# check running dockers
docker ps
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

```
docker volume create projects
docker run -it -v projects:/projects idaholab/moose:latest
cd /projects
moose-opt --copy-inputs phase_field
cd moose/phase_field
mkdir C2S
exit
# copy input file from local to docker

```

Run moose

```
# open docker
docker run -it -v projects://projects idaholab/moose:latest
# run moose
mpiexec -n 4 --allow-run-as-root moose-opt -i input.i
# exit docker
exit
```

<!-- ```
# open docker
docker run -it -v projects://projects idaholab/moose:latest
# run moose
mpiexec -n 4 --allow-run-as-root moose-opt -i input.i
# exit docker
exit
``` -->

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
- time = simulation time
- qclean sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)