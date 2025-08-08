# bib_ecc

This page documents the workflow for creating the input xml related to the `bib_ecc` model, i.e. model_bib_ecc.xml. The name `bib_ecc` stands for ball in box eccentric. The `bib` model represents a parallepiped nanoparticle of one SLD containing a spherical region of another SLD, shifted from the center.

Relevant input xml: model_bib_ecc.xml

Workflow:
1. [Copy model_bib_ecc.xml template](#copy-model_bib_eccxml-template)
2. [Edit model_bib_ecc.xml](#edit-model_bib_eccxml)
3. [Go back to main workflow](#go-to-workflow)

## Copy model_bib_ecc.xml template

```bash
# run from main folder
cd $C2S_HOME
# copy xml template to xml folder
cp xml/Template/model_bib_ecc.xml xml/
# open template with favourite editor (e.g. nano)
nano xml/model_bib_ecc.xml
```

## Edit model_bib_ecc.xml

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- ecc = vector from the center of the simulation box to the center of the spherical region
    - x = x value of vector
    - y = y value of vector
    - z = z value of vector
- qclean_sld = SLD outside of the simulation box

## Go to workflow

Go back to the main [workflow](../general/workflow.md#assign-sld-to-nodes) and continue.
