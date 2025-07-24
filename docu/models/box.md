# box

Description: Parallelepiped nanoparticle

Name: box

## Copy model_box.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_box.xml xml/
nano xml/model_box.xml
```

## Edit model_box.xml

- sld = SLD of parallelepiped
- qclean_sld = SLD outside simulation box

## Go to workflow

[Workflow](../general/workflow.md#assign-sld-to-nodes)