# bib

This page documents the workflow for creating the input xml related to the `bib` model, i.e. model_bib.xml. The name `bib` stands for ball in box. The `bib` model represents a parallepiped nanoparticle of one SLD containing a spherical region of another SLD at the center.

Relevant input xml: model_bib.xml

Workflow:
1. [Copy model_bib.xml template](#copy-model_bibxml-template)
2. [Edit model_bib.xml](#edit-model_bibxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_bib.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_bib.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_bib.xml
```

## Edit model_bib.xml

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of the parallelepiped nanoparticle
- qclean_sld = SLD outside of the simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
