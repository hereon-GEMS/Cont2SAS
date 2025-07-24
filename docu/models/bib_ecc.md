# bib_ecc

Description: Parallepiped nanoparticle with spherical region off the center

Name: ball in box eccentric (bib_ecc)

## Copy model_bib_ecc.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_bib_ecc.xml xml/
nano xml/model_bib_ecc.xml
```

## Edit model_bib_ecc.xml

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- ecc = vector w.r.t. simulation box center
    - x = x value of vector
    - y = y value of vector
    - z = z value of vector
- qclean_sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)