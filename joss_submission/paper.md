---
title: 'Cont2SAS: A python package for calculating SAS pattern from continuum structures'
tags:
  - Python
  - Continuum simulation
  - Small Angle Neutron Scattering
  - Small Angle X-ray Scattering
  - Sassena
authors:
  - name: Arnab Majumdar
    orcid: 0000-0003-4049-4060
    equal-contrib: true
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Martin M\"{u}ller
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2, 3"
  - name: Sebastian Busch
    orcid: 0000-0002-9815-909X
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: German Engineering Materials Science Centre (GEMS) at Heinz Maier-Leibnitz Zentrum (MLZ), Helmholtz-Zentrum Hereon GmbH, Lichtenbergstr. 1, 85748 Garching, Germany
   index: 1
 - name: Institute of Materials Physics, Helmholtz-Zentrum Hereon GmbH, Max-Planck-Str. 1, 21502 Geesthacht, Germany
   index: 2
 - name: Institut für Experimentelle und Angewandte Physik (IEAP), Christian-Albrechts-Universität zu Kiel, Leibnizstr. 19, 24098 Kiel, Germany
   index: 3
date: 17 July 2025
bibliography: paper.bib
---

# Summary

<!-- Continuum nanostructures can be created using physics-based simulations or knowledge-based generations. The simulated or generated nanostructures must be validated by comparing them with data obtained from different Small Angle Scattering (SAS) experiments, such as Small Angle Neutron Scattering (SANS) and Small Angle X-ray Scattering (SAXS). To perform the comparison,  -->

`Cont2SAS` software facilitates calculation of Small Angle Scattering (SAS) data from Continuum (Cont) nanostructures. `Cont2SAS` is built on the existing software solution `Sassena` -- known for calculating scattering patterns from atomic structures `[@lindner2012sassena; lindner2012towards; @sassena:2017, sassena:2023; majumdar2024computation]`. `Cont2SAS` can calculate: 
1. SAS patterns ($I$ vs $Q$: SAS intensity vs scattering vector magnitude) 
2. Time evolution of effective cross-section ($\sigma_{\text{eff}}$ vs $t$: neutron count rate per unit flux vs time)
  
Through the comparison of calculated and measured SAS data, simulations and SAS experiments can be used in a complementary manner to validate simulations, tune simulation parameters, and analyze SAS data obtained from experiments `[@dorrell2020combined; @reich2022comparison; @majumdar2024computation]`.

<!-- , are used to characterize material structure at the nanometer length scale. The experiments record Small Angle Scattering (SAS) intensity $(I)$ vs scattering vector magnitude $(Q)$ curves -- also at different times to capture structural change during a phenomena. To analyze the recorded data, nanostructures are generated or simulated, from which SAS patterns are calculated and compared with 

 The `Cont2SAS` software is used for calculating Small Angle Scattering (SAS) data numerically from continuum structures. The software is built on the existing software solution `Sassena` -- known for calculating scattering patterns from atomic structures. In addition to SAS patterns, `Cont2SAS` can also calculate effective-cross-section for comparing with neutron count rate. 
 The calculated SAS pattern from continuum structures are compared with measuremed data to validate simulations, tune simulation parametrs, and analyze SAS patterns `[@dorrell2020combined; @reich2022comparison; @majumdar2024computation]`.


 This method can help characterizing evolution of composition, which structure are not changing `[@dorrell2020combined; @reich2022comparison; @majumdar2024computation]`.

However, a direct retrieval of the material structure from a SAS pattern is not possible. So, SAS patterns are often calculated from material structures simulated based on physics or generated based on knwledge of the sample. The calculated patterns are compared with measured data.

continuum structures using the `Cont2SAS` software. Simulation of structure can be done based of physics, using other software, e.g. moose. The generation of structure can be performed based on knowledge of the sample using `Cont2SAS`. The calculated SAS pattern from continuum structures are compared with measuremed data to validate simulations, tune simulation parametrs, and analyze SAS patterns `[@dorrell2020combined; @reich2022comparison; @majumdar2024computation]`. -->

# Statement of need

The simulation of material structure at the nanometer length scale can be performed using atomistic simulations and continuum simulations. Atomistic simulations produce nanoscopic structures with atomic resolution, which makes them more accurate than continuum simutations that produce continuum structures. However, continuum simulations have an advantage over atomistic ones as continuum simulations can simulate bigger volumes for a larger time. 

The accuracy of continuum simulations can be checked by validating them against SAS experiments, such as Small Angle Neutron Scattering (SANS) and Small Angle X-ray Scattering (SAXS). The validation is performed by comparing SAS data calculated from simulated nanostrucres with measured ones. A validated simulation can also be used to retrieve nanostructure from SAS data because a direct retrieval of nanstructure from SAS data is not possible. 

Continuum nanostructures can also be generated based on knowledge of the sample, instead of simulating them based on physics based equations. Both generated and simulated structures are expected to output either Scattering Length Density (SLD) ($\beta$) values or a set of variables (e.g. density ($\rho$) and composition ($\chi$)) from which SLD values can be calculated. SAS patterns are calculated numerically from the SLD distribution using `Cont2SAS`.

In the `Cont2SAS` repository, 

Several examples of generated and simulated nanostructures are provided in the `Cont2SAS` repository. For simple geometries, scripts are provided that calculates SAS patterns numerically from generated structures and compares against analytical formula. One of these models does not change the structure but the chemical composition changes. For such a situation, the time evolution of neutron count rate can probe the composition change better than SAS patterns `[@aslan2019high]`. So, the corresponding scripts calculates time evolution of effective cross-section ($\sigma_{\text{eff}}$), which neutron count rate per unit neutron flux. The expected proportionality to the square of difference of SLD $((\Delta \beta)^2)$ is checked using the provided scripts. One example of simulated nanostructure is also provided for Fe- Cr spinodal decomposition. The simulation is performed using moose framework. Script are provided that convert the moose output to a series of hdf files, which are then read by `Cont2Sas` package for assigning SLD. A script is also provided that calculates the SAS pattern from FEM simulation of Fe-Cr spinodal decomposition.

`Cont2SAS` provides the much needed software platform for calculating SAS pattern from continuum simulations. The addition of effective cross-section in the software package is going be helpful for analyzing powder-like structures `[@aslan2019high]`. One can also retrieve continuum nanostructures from SAS data using generated or simulated model strcutures.  

<!-- The numerical calculations are validated

A direct retieval of the nanosctructure from SAS data is not possible due to under-determination. So, SAS data is calculated from simulated nanostrucres and compared to measured data. Such 

Once the simulation method is validated, this

A validated simulation can also be used for anal


from can also be analyzed using simulations as a direct retrival of nanostructure is not possible from SAS data


Since the SAS experients cover a large volume and can probe nanoscopic phenomema for a time scale in the range of seconds, continuum simulations are sometimes the only simulation method that can be used to


atomistic simulations are limited in length and time scales. Whereas, continuum simulations can produce nanostructures that cover bigger volumes for larger times. So, the continuum simulations, validated against experiments, can be used to simulate big volumes for sugfficient time

A bigger volume simulatedDifferent SAS techniques


 and can simulate with sufficient accuracy provide a better alternative 

Since SAS intensities contain information of a macroscopic volume and can probe nanoscopic phenomena occuring in the range of milliseconds, Continuum simulations are better alternative for 

Different simulation methodologies 

The complementary use of simulations and SAS patterns can be performed with different simulations methodogies, which can broadly be classified into two categories: Atomistic simulations and continuum simulations. The atomistic simulations produce nanoscopic structures with atomic resolution, from which SAS pattern can be calculated in a fast and error-free manner using the software solution `Sassena` `[@lindner2012sassena; @lindner2012towards; @sassena:2017; @sassena:2023; @majumdar2024computation]`. The continuum simulation produces nanoscopic structures as continuous distribution of density and composition. To calculate SAS patterns from these continuous distributions, a new software was required that facilitates fast and error-free calculations. 

`Cont2SAS` was created for calculating SAS patterns from continuum structures, realised through the spatial distribution of Scattering Length Density (SLD). The SLD distribution can either be calculated from local density and composition obtained from physics-based simulations or be generated based on knowledge of the sample. The generated or simulated structure is taken as an input to `Cont2SAS`, which is then discretized to an atom-like structure, from which SAS patterns are calculated using `Sassena`. Using Sassena, ensures fast and error-free calculation `[@majumdar2024computation]`.

From SANS patterns, a new quantity named `effective cross-section` can also be calculated using `Cont2SAS`. `effective cross-section` can be understood as neutron count rate per unit neutron flux. Comparing `effective cross-section` and `neutron count rate` can be usefull for samples that does not change its nanoscopic structure during a process, but the chemical composition is changed `[@aslan2019high]`.

The integration of continuum simulation and SAS measurements opens the door for simulating materials for bigger sample volumes for longer times. This is an essential improvement because the SAS technique also probes macroscopic volumes for time scale in the rage in seconds. -->

# Mathematics
Calculation of SLD from local density and composition:
$$
\beta(\vec{r})= \rho_{\text{m}}(\vec{r}) \cdot N_{\text{A}} \cdot \sum_{\alpha} \chi_{\alpha}(\vec{r}) \cdot \sum_{\text{el}} c_{\alpha, \text{el}} \cdot b_{\text{el}}
$$

Numerical calculation of SAS pattern from discretized SLD distribution:
$$
I(Q)=\left\langle\left|\sum_{j=1}^{N}\sum_{k=1}^{N} b_j b_k e^{\mathrm{i} \vec{Q} \cdot (\vec{R}_j -\vec{R}_k)}\right|^2\right\rangle_{\Omega}
$$

Analytical calculation of SAS pattern from SLD distribution:
$$
I(Q)=\left\langle\left|\iiint\limits_V \beta(\vec{r})\,\mathrm{e}^{\mathrm{i} \vec{Q} \cdot \vec{r}}\,\mathrm{d}\vec{r}\,\right|^2\right\rangle_{\Omega}
$$

Numerical calculation of effctive cross-section $\sigma_{\text{eff}}$ from SAS pattern:
$$
\sigma_{\text{eff}}=\int_{Q}w(Q)I(Q) \propto (\Delta \beta)^2
$$





# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

This publication was written in the context of the work of the consortium DAPHNE4NFDI in
association with the German National Research Data Infrastructure (NFDI) e.V. NFDI is financed by
the Federal Republic of Germany and the 16 federal states and the consortium is funded by the
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number
460248799. The authors would like to thank for the funding and support. Furthermore, thanks go to
all institutions and actors who are committed to the association and its goals.

# References