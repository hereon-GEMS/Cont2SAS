# box

This page documents the workflow for creating the input xml related to the `box` model, i.e. model_box.xml. The `box` model represents a parallelepiped nanoparticle surrounded by vaccum.

Relevant input xml: model_box.xml

Workflow:
1. [Copy model_box.xml template](#copy-model_boxxml-template)
2. [Edit model_box.xml](#edit-model_ballxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_box.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_box.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_box.xml
```

## Edit model_box.xml

- sld = SLD of parallelepiped
- qclean_sld = SLD outside of the simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
