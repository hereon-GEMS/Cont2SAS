---
title: 'Cont2SAS: A Python package for calculating small angle scattering parameters from continuum nanostructures'
tags:
  - Python
  - Continuum simulation
  - Small Angle Neutron Scattering
  - Small Angle X-ray Scattering
  - Sassena
authors:
  - name: Arnab Majumdar
    orcid: 0000-0003-4049-4060
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Martin Müller
    orcid: 0000-0003-0995-7602
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

`Cont2SAS` is a software tool built around the existing software solution `Sassena` -- known for calculating scattering patterns from simulated atomic structures. The goal of `Cont2SAS` is to provide software similar to `Sassena`, but for calculating Small Angle Scattering (SAS) data from simulated continuum nanostructures. `Cont2SAS` can calculate SAS patterns, i.e. SAS intensity $(I)$ vs. scattering vector magnitude $(Q)$ using a numerical method. `Cont2SAS` can also calculate the effective cross-section $(\sigma_{\text{eff}})$, i.e. the count rate of scattered radiation per incident unit flux. $\sigma_{\text{eff}}$ is calculated by integrating instrument-agnostic SAS patterns taking the instrument geometry into account. `Cont2SAS` can be used for different purposes, such as validating simulations, tuning simulation parameters, and analyzing SAS data.

# Statement of need

SAS experiments with neutrons (SANS) and X-rays (SAXS) are useful techniques for probing material nanostructures [@chen2012characterizing]. With the Time Resolved (TR) variant of SAS, the time evolution of nanostructures can also be probed [@hollamby2013practical]. However, a direct retrieval of the nanostructure from SAS data is not possible due to the so-called phase problem, i.e. the phase loss in measured intensities [@billinge2007problem]. Therefore, it is fruitful to combine SAS data with simulations to study nanostructures [@majumdar2024computation]. This approach is also useful for validating theories underlying simulations [@reich2022comparison].

Simulated real-space structures can be generated using atomistic or continuum simulations. However, atomistic simulations cannot feasibly simulate large structures (e.g. micrometer scale) over long times (e.g. seconds) due to computational limitations [@ghavanloo2019computational]. This becomes problematic when nanostructure evolution is influenced by larger length-scale structures or when the TR-SAS data is recorded over long time periods. In such cases, continuum simulation is an appropriate option. `Cont2SAS` aims to provide a tool that offers fast comparison of continuum simulations and SAS data. The ultimate goal is to study nanomaterials and validate theories using SAS data, particularly when length-scales and time-scales, inaccessible to atomistic simulations, need to be simulated.

# State of the field

Different software packages such as `nMOLDYN` [@kneller1995nmoldyn; @rog2003nmoldyn; @calandrini2011nMoldyn; @hinsen2012nMoldyn3; @nmoldyn2023hinsen], `MDANSE` [@goret2017mdanse; @mdanse2024agu], `LiquidLib` [@walter2018liquidlib; @liquidlib2023zlab], and `Sassena` [@lindner2012sassena; @lindner2012towards; @sassena2023majumdar; @majumdar2024computation] can calculate SAS patterns from atomistic simulations, but they cannot calculate the same from continuum simulations. For continuum simulations, existing software implementations are typically developed for problem-specific applications [@schmidt2007simulation; @dorrell2020combined; @berbenni2025simulation]. The `Generic SAS Calculator` option in `SasView` [@sasview] is also a potential alternative. However, `SasView` doesn't provide an option to convert simulation output into either discretized points or elements with constant Scattering Length Density (SLD) required by the `Generic SAS Calculator` [@liu2023generic]. Hence, `Cont2SAS` is built as a general-purpose program that converts simulation output into discretized atoms using a well-defined strategy [@majumdar2026characterization]. For calculating SAS data from the discretized atoms, `Cont2SAS` uses `Sassena` as a backend calculator, which provides notable computation speed and robustness [@majumdar2024computation].

# Software features

![Workflow of calculating SAS patterns: [left] SLD assignment for a spherical nanoparticle on generated mesh and [right] numerical calculation of SAS pattern. The numerical calculation is validated against the known analytical formula for spherical nanoparticles [@guinier1955small]. \label{fig:sas_workflow}](figures/workflow.png)

`Cont2SAS` calculates SAS patterns taking simulated nanostructures as input. The simulated structure must include a SLD distribution, which is either directly obtained from simulation or calculated from a set of simulated variables (e.g. molar density and molar fraction) before being provided as input to `Cont2SAS`. Simulating SLD distributions is also possible within `Cont2SAS` for model structures.  The simulated input is processed to data tailor-made for `Sassena`, which calculates SAS patterns, i.e. $I$ vs $Q$ data, as shown in \autoref{fig:sas_workflow}. The $Q$-dependence of $I$ reveals the average shape and size of the nanoparticles, whereas the magnitude of $I$ is related to the scattering contrast, i.e. the average difference in SLD between particles and their surrounding medium.

![[left] TR-SAS data, i.e. SAS patterns at different time steps ranging from 0 to 10 seconds. The SLD of a spherical nanoparticle increases linearly with time. [right] $\sigma_{\text{eff}}$ calculated from the SAS patterns. The calculated $\sigma_{\text{eff}}$ values are proportional to the square of the SLD difference between the particle and its environment, i.e. ($\Delta$SLD)$^2$, whose shown values are multiplied by an empirical constant. \label{fig:sig_eff_workflow}](figures/sig_eff.png)

For simulations with multiple time steps, SAS patterns can be calculated from different time steps generating a TR-SAS dataset, as shown in \autoref{fig:sig_eff_workflow}. From TR-SAS data, `Cont2SAS` can calculate the time evolution of the effective cross-section $(\sigma_{\text{eff}})$, which is defined as the count rate per incident unit flux (see \autoref{fig:sig_eff_workflow}). This feature was not available in previous implementations [@schmidt2007simulation; @dorrell2020combined; @liu2023generic; @berbenni2025simulation] or `Sassena` [@sassena2023majumdar]. The time evolution of the count rate is useful when the SLD of the nanostructure changes over time without any change in average shape and size, e.g. while storing hydrogen in a ball-milled powder sample [@aslan2019high]. The calculated $\sigma_{\text{eff}}$ must be multiplied by an empirical factor before comparing it with the measured count rate.   

# Conclusion

`Cont2SAS` provides a software package for calculating SAS patterns from continuum simulations of nanostructures. The addition of the effective cross-section in the software package further enables the analysis of the experimental count rate. `Cont2SAS` can be used for analyzing SAS data and validating simulations to study nanomaterials.

# Acknowledgements

This publication was written in the context of the work of the consortium DAPHNE4NFDI in association with the German National Research Data Infrastructure (NFDI) e.V. NFDI is financed by the Federal Republic of Germany and the 16 federal states and the consortium is funded by the
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248799. The authors would like to thank for the funding and support. Furthermore, thanks go to all institutions and actors who are committed to the association and its goals.

# Declaration

During the preparation of this work, the authors used ChatGPT in order to improve the quality of the code and the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published code and article.

# References