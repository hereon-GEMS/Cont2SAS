# bib

Description: Parallepiped nanoparticle with spherical region at the center

Name: ball in box (bib)

## Copy model_bib.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_bib.xml xml/
nano xml/model_bib.xml
```

## Copy model_bib.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_bib.xml xml/
nano xml/model_bib.xml
```

## Edit model_bib.xml

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)