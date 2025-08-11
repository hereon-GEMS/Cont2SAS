# Quick start guide

## List of models

Provided models either generate scattering length density (SLD) values based on analytical formulae or simulate the SLD distribution by solving physics-based simulations (e.g. a phase-field simulation). Some of the generated models produce a structure at a single time step; others produce time-evolving structures, representing phenomena.

The following models are included in this repository, the instructions how to run them can be found below:

- Generated models
    - Structure
        - [Sphere (ball)](#ball)
        - [Cube (box)](#box)
        - [Sphere at the center of cube (ball in box, bib)](#bib)
        - [Sphere off the center of cube (eccentric ball in box, bib_ecc)](#bib_ecc)
    - Phenomena
        - [Growth of sphere (gg)](#gg)
        - [Interdiffusion of sphere and environment (fs)](#fs)
        - [Change of chemical composition of sphere (sld_growth)](#sld_growth)
- Simulated models
    - Phase field
        - [Spinodal decomposition of Fe-Cr (phase_field)](#phase_field)

Instructions how to run these models are given in this document below; a [detailed documentation of each model and the complete workflow](./docu/index.md) is also available.

Follwing are computation time required by generation scripts of each model in three different machines (M1, M2 - Dell XPS 13 Laptop (2017), M3):

| Model       | Runtime - M1 | Runtime - M2 | Runtime - M3 | Avg runtime|
|-------------|-------------:|-------------:|-------------:|-----------:|
| ball        |         12 s |          9 s |         xx s |       xx s |
| box         |         28 s |         32 s |         xx s |       xx s |
| bib         |         28 s |         31 s |         xx s |       xx s |
| bib_ecc     |         40 s |         44 s |         xx s |       xx s |
| gg          |         48 s |         56 s |         xx s |       xx s |
| fs          |       2426 s |       2603 s |         xx s |       xx s |
| sld_grow    |         54 s |         66 s |         xx s |       xx s |
| phase_field |        882 s |       1008 s |         xx s |       xx s |

## ball

A sphere surrounded by empty space.
Run time approx. 1 minute.

```bash
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

## box

A cube surrounded by empty space.
Run time approx. 1 minute.

```bash
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

## bib

A sphere at the center of a cube.
Run time approx. 1 minute.

```bash
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

## bib_ecc

A sphere inside a cube, displaced from the center.
Run time approx. 1 minute.

```bash
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

## gg

A spherical grain growing in size from 7 to 17$\AA$ over 11 time steps.
Run time approx. 1 minute.

```bash
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

## fs

Interdiffusion of a spherical inhomogeneity with its environment over 11 time steps.
Run time approx. 1 hour.

```bash
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

## sld_grow

Change of the chemical composition of a sphere over 11 time steps.
Run time approx. 1 minute.

```bash
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

## phase_field

Phase-field simulation of the spinodal decomposition of a Fe-Cr mixture.
Run time approx. 30 minutes.

```bash
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

## Link to README

[README](./README.md)
