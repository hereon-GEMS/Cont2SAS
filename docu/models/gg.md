# gg

Description: Growth of spherical grain over time

Name: grain growth (gg)

## Copy model_gg.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_gg.xml xml/
nano xml/model_gg.xml
```

## Edit model_gg.xml

- rad_0 = starting radius of sphere
- rad_end = end radius of sphere
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)