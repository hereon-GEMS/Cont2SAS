# ball

The `ball` model represents a spherical nanoparticle surrounded by vaccum.

Relevant input xml: model_ball.xml

## Copy model_ball.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_ball.xml xml/
nano xml/model_ball.xml
```

## Edit model_ball.xml

- rad = radius of sphere
- sld = SLD of sphere
- qclean_sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)