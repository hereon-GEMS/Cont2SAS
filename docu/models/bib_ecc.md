# bib_ecc

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