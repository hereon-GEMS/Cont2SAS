# phase_field
Description: Iron - Chromium spinodal decomposition

Name: phase_field

## Copy model_ball.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_ball.xml xml/
nano xml/model_ball.xml
```

## Edit model_ball.xml

- name = name of the model (= 'spinodal_fe_cr')
- time = simulation time
- qclean sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)