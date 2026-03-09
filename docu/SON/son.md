# Statement of need

Small Angle Scattering (SAS) experiments with neutrons (SANS) and X-rays (SAXS) are useful techniques for probing material nanostructures [1]. With the Time Resolved (TR) variant of SAS, the time evolution of nanostructures can also be probed [2]. However, a direct retrieval of nanostructure from SAS data is not possible due to the so-called phase problem, i.e. the phase loss in measured intensities [3]. Therefore, it is fruitful to combine SAS data with simulations to study nanostructures [4]. This approach is also useful for validating theories underlying simulations [5].

Simulated real-space structures can be generated using atomistic or continuum simulation. However, atomistic simulation cannot simulate large structures (e.g. millimeter scale) over long times (e.g. seconds) due to computational limitations [6]. This becomes problematic when the nanostructure evolution is influenced by larger length-scale structures or when the SAS data is recorded over long time periods. In such cases, continuum simulation is the only option. Despite this indispensability, there is no software available for calculating SAS data from continuum simulations. `Cont2SAS` aims to fill this void by providing a tool that offers fast and error-free comparison of continuum simulations and SAS data. The ultimate goal is to study nanomaterials and validating theories using SAS data, particularly when length-scales and time-scales, inaccessible to atomistic simulations, need to be simulated.

## Reference

[1] Chen, Z. H., Kim, C., Zeng, X., Hwang, S. H., Jang, J., & Ungar, G. (2012). Characterizing size and porosity of hollow nanoparticles: SAXS, SANS, TEM, DLS, and adsorption isotherms compared. Langmuir, 28(43), 15350–15361. https://doi.org/10.1021/la302236u

[2] Hollamby M. J. (2013) Practical applications of small-angle neutron scattering. Phys. Chem. Chem. Phys., 15, 10566–10579. https://doi.org/10.1039/C3CP50293G

[3] Billinge, S. J., & Levin, I. (2007). The problem with determining atomic structure at the nanoscale. Science, 316(5824), 561–565. https://doi.org/10.1126/science.1135080

[4] Majumdar A., Müller, M., & Busch, S. (2024). Computation of x-ray and neutron scattering patterns to benchmark atomistic simulations against experiments. International Journal of Molecular Sciences, 25(3), 1547. https://doi.org/10.3390/ijms25031547

[5] Reich, V., Majumdar, A., Müller, M., & Busch, S. (2022). Comparison of molecular dynamics simulations of water with neutron and x-ray scattering experiments. EPJ Web of Conferences, 272, 01015. https://doi.org/10.1051/epjconf/202227201015

[6] Ghavanloo, E., Rafii-Tabar, H., & Fazelzadeh, S. A. (2019). Computational continuum mechanics of nanoscopic structures. Springer. https://doi.org/10.1007/978-3-030-11650-7