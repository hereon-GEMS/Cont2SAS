---
title: 'Cont2SAS: A python package for calculating SAS parameters from continuum nanostructures'
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

`Cont2SAS` is a software tool built around the existing software solution `Sassena` -- known for calculating scattering patterns from simulated atomic structures. The goal of `Cont2SAS` is to provide a software similar to `Sassena`, but for calculating Small Angle Scattering (SAS) data from simulated Continuum (Cont) nanostructures. `Cont2SAS` can calculate SAS patterns, i.e. SAS intensity $(I)$ vs. scattering vector magnitude $(Q)$ using a novel numerical method. `Cont2SAS` can also calculate effective cross-section $(\sigma_{\text{eff}})$, i.e. the count rate of scattered radiation per incident unit flux. $\sigma_{\text{eff}}$ is calculated by integrating instrument-agnostic SAS patterns taking instrument geometry into account. `Cont2SAS` can be used for different purposes, such as validating simulations, tuning simulation parameters, and analyzing SAS data.

# Statement of need

Small Angle Scattering (SAS) experiments with neutrons (SANS) and X-rays (SAXS) are useful techniques for probing material nanostructures [@chen2012characterizing]. With the Time Resolved (TR) variant of SAS, the time evolution of nanostructures can also be probed [@hollamby2013practical]. However, a direct retrieval of nanostructure from SAS data is not possible due to the so-called phase problem [@billinge2007problem]. Therefore, it is fruitful to combine SAS data with simulations to study nanostructures [@majumdar2024computation]. This approach is also useful for validating theories underlying simulations [@reich2022comparison].

Simulated structures in real space can be generated using atomistic and continuum simulation. However, continuum simulation is the only option when the nanostructure evolution is influenced by bigger features or the SAS data is recorded for long time period, as atomistic simulation can not simulate large structures for long time [@ghavanloo2019computational]. Despite this indispensability, there is no software available for calculating SAS data from continuum simulations. `Cont2SAS` aims to fill this void by providing a tool that offers fast and error-free comparison of continuum simulations and SAS data. The ultimate goal is to study nanomaterials and validating theories using SAS data, particularly when length-scales and time-scales, inaccessible to atomistic simulations, need to be simulated.

# State of the field

From atomistic simulations, different software such as `nMoldyn` [@rog2003nmoldyn; @nmoldyn2023hinsen], `MDANSE` [@goret2017mdanse; @mdanse2023isis], `LiquidLib` [@walter2018liquidlib; @liquidlib2023zlab], and `Sassena` [@lindner2012sassena; @lindner2012towards; @sassena2017lindner; @sassena2023majumdar; @majumdar2024computation] can calculate SAS patterns but they can not calculate the same from continuum simulations. An in-house code is available for conitnuum simulations but it was built for a 2D membrane system lacking generality [@dorrell2020combined]. Hence, `Cont2SAS` is built using `Sassena` as a backend calculator. `Sassena` was chosen over other software due to its notable computation speed and robustness [@majumdar2024computation].

# Software features

![Workflow of SAS pattern calculation: [left] SLD assignment for a spherical nanoparticle on generated mesh and [right] numerical calculation of SAS pattern. The numerical calculation is validated against known analytical formula for spherical nanoparticles [@guinier1955small]. \label{fig:sas_workflow}](figures/workflow.png)

`Cont2SAS` calculates SAS patterns taking simulated nanostructures as an input. The simulated structure must provide either Scattering Length Density (SLD) ($\beta$) values or a set of variables from which SLD values can be calculated. The simulated input is processed to a data taylor-made for `Sassena`. `Sassena` calculates the SAS intensity $(I)$ as a function of scattering vector magnitude $(Q)$, i.e. SAS pattern, for different time steps (see \autoref{fig:sas_workflow}). 

![Workflow of effective cross-section $(\sigma_{\text{eff}})$ calculation: [left] SAS patterns at different timesteps ranging from 0 to 10 seconds and [right] $\sigma_{\text{eff}}$ calculated from SAS patterns. With time, the SLD of a spherical nanoparticle increases linearly. The calculated $\sigma_{\text{eff}}$ values are proportional to the square of SLD difference between the particle and its environment, shown as contrast. \label{fig:sig_eff_workflow}](figures/sig_eff.png)

`Cont2SAS` can also calculate the time evolution of effective cross-section $(\sigma_{\text{eff}})$, which is defined as the count rate per incident unit flux. This feature was not available in the in-house code [@dorrell2020combined] or `Sassena` [@sassena2023majumdar]. The time evolution of count rate is useful when the chemical composition changes over time instead of the nanostructure, e.g. while storing hydrogen in ball-milled powder sample [@aslan2019high]. \autoref{fig:sig_eff_workflow} demonstrates the workflow of such calculation from a series of SAS patterns. The calculated $\sigma_{\text{eff}}$ must be multiplied by an empirical factor before comparing with measured count rate.   

# Conclusion

`Cont2SAS` provides a software platform for calculating SAS pattern from continuum simulations of nanostructures. The addition of effective cross-section in the software package further enables the analysis of count rate. `Cont2SAS` can be used for analyzing SAS data and validating simulations to study nanomaterials.

# Acknowledgements

This publication was written in the context of the work of the consortium DAPHNE4NFDI in association with the German National Research Data Infrastructure (NFDI) e.V. NFDI is financed by the Federal Republic of Germany and the 16 federal states and the consortium is funded by the
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248799. The authors would like to thank for the funding and support. Furthermore, thanks go to all institutions and actors who are committed to the association and its goals.

# Declaration

During the preparation of this work, the authors used Chat-GPT in order to improve the quality of code and article. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published code and article.

# References