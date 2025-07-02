## Model structures

### ball

Description: Spherical nanoparticle</br>
Name: ball

Structure of ``model_ball.xml``:

- rad = radius of sphere
- sld = SLD of sphere
- qclean_sld = SLD outside simulation box

### box

Description: Parallelepiped naniparticle</br>
Name: box

Structure of ``model_box.xml``:

- sld = SLD of parallelepiped
- qclean_sld = SLD outside simulation box

### bib

Description: Parallepiped nanoparticle with spherical region at the center</br>
Name: ball in box (bib)

Structure of ``model_bib.xml``:

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### bib_ecc

Description: Parallepiped nanoparticle with spherical region off the center</br>
Name: ball in box eccentric (bib_ecc)

Structure of ``model_box.xml``:

- rad = radius of sphere (must be <= simulation box length/2)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- ecc = vector w.r.t. simulation box center
    - x = x value of vector
    - y = y value of vector
    - z = z value of vector
- qclean_sld = SLD outside simulation box

### gg

Description: Growth of spherical grain over time</br>
Name: grain growth (gg)

Structure of ``model_gg.xml``:

- rad_0 = starting radius of sphere
- rad_end = end radius of sphere
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### fs

Description: Interdiffusion between spherical grain and its environment over time</br>
Name: fuzzy sphere (fs)

Structure of ``model_fs.xml``:

- rad = sphere radius
- sig_0 = starting fuzz value (>=0)
- sig_end = end fuzz value (>=0)
- sld_in = SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box

### sld_grow

Description: Change of chemical composition of spherical grain over time</br>
Name: sld growth (sld_grow)

Structure of ``model_sld_grow.xml``:

- rad = sphere radius
- sld_in_0 = starting SLD of sphere
- sld_in_end = end SLD of sphere
- sld_out = SLD of environment
- qclean_sld = SLD outside simulation box


## FEM simulation

### phase_field
Description: Iron - Chromium spinodal decomposition

- name = name of the model (= 'spinodal_fe_cr')
- time = simulation time
- qclean sld = SLD outside simulation box