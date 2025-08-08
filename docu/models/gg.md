# gg

This page documents the workflow for creating the input xml related to the `gg` model, i.e. model_gg.xml. The name `gg` stands for grain growth. The `gg` model represents the phenomenon of the growth of a spherical grain in a cubic nanoparticle over time.

Relevant input xml: model_gg.xml

Workflow:
1. [Copy model_gg.xml template](#copy-model_ggxml-template)
2. [Edit model_gg.xml](#edit-model_ggxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_gg.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_gg.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_gg.xml
```

## Edit model_gg.xml

- rad_0 = starting radius of sphere
- rad_end = end radius of sphere
- sld_in = SLD of sphere
- sld_out = SLD of the cubic nanoparticle
- qclean_sld = SLD outside of the simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
