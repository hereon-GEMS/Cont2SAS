# Quick start guide

## List of models

Provided models either generates SLD based on analytical formula or simulates SLD solving physics-based simulations (e.g. phase-field simulation). Within generated models, some produces structure at one time step; some produces time-evolving structures representing phenomena.

Following are the list of models:

- Generated models
    - Structure
        - Sphere (ball)
        - Cube (box)
        - Sphere at the center of cube (bib)
        - Sphere off the center of cube (bib_ecc)
    - Phenomena
        - Growth of sphere (gg)
        - interdiffusion of sphere and environment (fs)
        - Change of chemical composition of sphere (sld_growth)
- Simulated models
    - Phase field
        - Spinodal decomposition of Fe-Cr (phase_field)

## ball

### Instruction

```
# generate data
python models/ball/ball_gen.py
# generate plots
python models/ball/ball_plot.py
# see nano structure
xdg-open figure/ball/SLD_ball.pdf
# see discretized nano structure
xdg-open figure/ball/pseudo_ball.pdf
# see categorized nano structure
xdg-open figure/ball/pseudo_cat_ball.pdf
# see SAS intensity vs Q plot
xdg-open figure/ball/Iq_ball.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## box

### Instruction

```
# generate data
python models/box/box_gen.py
# generate plots
python models/box/box_plot.py
# see nano structure
xdg-open figure/box/SLD_box.pdf
# see discretized nano structure
xdg-open figure/box/pseudo_box.pdf
# see categorized nano structure
xdg-open figure/box/pseudo_cat_box.pdf
# see SAS intensity vs Q plot
xdg-open figure/box/Iq_box.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## bib

### Instruction

```
# generate data
python models/bib/bib_gen.py
# generate plots
python models/bib/bib_plot.py
# see nano structure
xdg-open figure/bib/SLD_bib.pdf
# see discretized nano structure
xdg-open figure/bib/pseudo_bib.pdf
# see categorized nano structure
xdg-open figure/bib/pseudo_cat_bib.pdf
# see SAS intensity vs Q plot
xdg-open figure/bib/Iq_bib.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## bib_ecc

### Instruction

```
# generate data
python models/bib_ecc/bib_ecc_gen.py
# generate plots
python models/bib_ecc/bib_ecc_plot.py
# see nano structure
xdg-open figure/bib_ecc/SLD_bib_ecc.pdf
# see discretized nano structure
xdg-open figure/bib_ecc/pseudo_bib_ecc.pdf
# see categorized nano structure
xdg-open figure/bib_ecc/pseudo_cat_bib_ecc.pdf
# see SAS intensity vs Q plot
xdg-open figure/bib_ecc/Iq_bib_ecc.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## gg

### Instruction

```
# generate data
python models/gg/gg_gen.py
# generate plots
python models/gg/gg_plot.py
# see nano structure (t = 1.0)
xdg-open figure/gg/SLD_gg_1p0.pdf
# see discretized nano structure (t = 1.0)
xdg-open figure/gg/pseudo_gg_1p0.pdf
# see categorized nano structure (t = 1.0)
xdg-open figure/gg/pseudo_cat_gg_1p0.pdf
# see SAS intensity vs Q plot
xdg-open figure/gg/Iq_gg.pdf
# see simulation vs retrieved radius
xdg-open figure/gg/rad_fit_gg.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## fs

### Instruction

```
# generate data
python models/fs/fs_gen.py
# generate plots
python models/fs/fs_plot.py
# see nano structure (t = 1.0)
xdg-open figure/fs/SLD_fs_1p0.pdf
# see discretized nano structure (t = 1.0)
xdg-open figure/fs/pseudo_fs_1p0.pdf
# see categorized nano structure (t = 1.0)
xdg-open figure/fs/pseudo_cat_fs_1p0.pdf
# see SAS intensity vs Q plot
xdg-open figure/fs/Iq_fs.pdf
# see simulation vs retrieved radius
xdg-open figure/fs/rad_fit_fs.pdf
# see simulation vs retrieved sigma value
xdg-open figure/fs/sig_fit_fs.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## sld_grow

### Instruction

```
# generate data
python models/sld_grow/sld_grow_gen.py
# generate plots
python models/sld_grow/sld_grow_plot.py
# see nano structure (t = 1.0)
xdg-open figure/sld_grow/SLD_sld_grow_1p0.pdf
# see discretized nano structure (t = 1.0)
xdg-open figure/sld_grow/pseudo_sld_grow_1p0.pdf
# see categorized nano structure (t = 1.0)
xdg-open figure/sld_grow/pseudo_cat_sld_grow_1p0.pdf
# see SAS intensity vs Q plot
xdg-open figure/sld_grow/Iq_sld_grow.pdf
# see simulation vs retrieved sld value
xdg-open figure/sld_grow/sld_fit_sld_grow.pdf
# see effective c.s. prop to contrast square
xdg-open figure/sld_grow/sig_eff_fit_sld_grow.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## phase_field

### Instruction

```
# generate data
python models/phase_field/phase_field_gen.py
# generate plots
python models/phase_field/phase_field_plot.py
# see nano structure (t = 86400)
xdg-open figure/phase_field/SLD_phase_field_86400.pdf
# see discretized nano structure (t = 86400)
xdg-open figure/phase_field/pseudo_phase_field_86400.pdf
# see categorized nano structure (t = 86400)
xdg-open figure/phase_field/pseudo_cat_phase_field_86400.pdf
# see SAS intensity vs Q plot
xdg-open figure/phase_field/Iq.pdf
# see characteristic length vs t plot
# initial step eliminated
xdg-open figure/phase_field/ch_len.pdf
```

### See detailed explanation

- [Link to documentation](./docu/index.md)

## Link to README

[README](./README.md)