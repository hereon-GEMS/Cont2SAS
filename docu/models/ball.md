# ball

This page documents the workflow for creating the input xml related to the `ball` model, i.e. model_ball.xml. The `ball` model represents a spherical nanoparticle surrounded by vaccum.

Relevant input xml: model_ball.xml

Workflow:
1. [Copy model_ball.xml template](#copy-model_ballxml-template)
2. [Edit model_ball.xml](#edit-model_ballxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_ball.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_ball.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_ball.xml
```

## Edit model_ball.xml

- rad = radius of sphere
- sld = SLD of sphere
- qclean_sld = SLD outside of the simulation box

## Go to workflow
Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
