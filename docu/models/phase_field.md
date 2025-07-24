# Phase field
Description: Iron - Chromium spinodal decomposition

Name: phase_field

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

```

```
docker volume create projects
docker run -it -v projects:/projects idaholab/moose:latest
cd /projects
moose-opt --copy-inputs phase_field
cd moose/phase_field
mkdir C2S
exit
```

```
# open docker
docker run -it -v projects://projects idaholab/moose:latest
# run moose
mpiexec -n 4 --allow-run-as-root moose-opt -i input.i

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
- time = simulation time
- qclean sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)