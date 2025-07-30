# fs

This page documents the workflow for creating the input xml related to the `fs` model, i.e. model_fs.xml. The name `fs` stands for fuzzy sphere. The `fs` model represents the interdiffusion phenomena between spherical grain and its environment over time. The model is named fuzzy sphere because the resulting model is fitted to the analytical formula of fuzzy sphere.

Relevant input xml: model_fs.xml

Workflow:
1. [Copy model_fs.xml template](#copy-model_fsxml-template)
2. [Edit model_fs.xml](#edit-model_fsxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_fs.xml template

```
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_fs.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_fs.xml
```

## Edit model_fs.xml

Structure of ``model_fs.xml``:

- rad = sphere radius
- sig_0 = starting fuzz value (>=0)
- sig_end = end fuzz value (>=0)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.