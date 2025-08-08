# sld_grow

This page documents the workflow for creating the input xml related to the `sld_grow` model, i.e. model_sld_grow.xml. The `sld_grow` model represents the phenomenon of a changing chemical composition of a spherical grain inside a cubic nanoparticle over time. The effective cross-section is also calculated for the `sld_grow` model.

Relevant input xml: model_sld_grow.xml

## Copy model_sld_grow.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_sld_grow.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_sld_grow.xml
```

## Edit model_ball.xml

- rad = sphere radius
- sld_in_0 = starting SLD of sphere
- sld_in_end = end SLD of sphere
- sld_out = SLD of the cubic nanoparticle
- qclean_sld = SLD outside of the simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
