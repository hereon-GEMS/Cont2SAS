# fs

Description: Interdiffusion between spherical grain and its environment over time

Name: fuzzy sphere (fs)

## Copy model_fs.xml template

```
# run from main folder
# relative position ../
cd $C2S_HOME
cp xml/Template/model_fs.xml xml/
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

[Workflow](../general/workflow.md#assign-sld-to-nodes)