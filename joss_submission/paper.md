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

`Cont2SAS` facilitates the calculation of Small Angle Scattering (SAS) parameters from simulated Continuum (Cont) nanostructures. `Cont2SAS` is built on the existing software solution `Sassena` -- known for calculating scattering patterns from simulated atomic structures [@lindner2012sassena; @lindner2012towards; @sassena2017lindner; @sassena2023majumdar; @majumdar2024computation]. `Cont2SAS` can calculate SAS patterns and the effective scattering cross-section $(\sigma_{\text{eff}})$. SAS patterns contain a SAS intensity $(I)$ vs. scattering vector magnitude $(Q)$. The $\sigma_{\text{eff}}$ is the count rate of scattered radiation per incident unit flux. The time evolution of $\sigma_{\text{eff}}$ is calculated from SAS patterns at different time steps. Through the comparison of calculated and measured SAS parameters, simulations and SAS experiments can be used complementarily for different purposes, such as validating simulations, tuning simulation parameters, and analyzing SAS data obtained from experiments [@dorrell2020combined; @reich2022comparison; @majumdar2024computation].

# Statement of need

The simulation of material structure at the nanometer length scale can be performed using atomistic simulations and continuum simulations. Continuum simulations have the advantage over atomistic ones that they can simulate bigger volumes for a larger time. However, continuum simulations are less accurate than the atomistic simulations. `Cont2SAS` is created to check the accuracy of continuum simulations by validating them against SAS experiments, such as Small Angle Neutron Scattering (SANS) and Small Angle X-ray Scattering (SAXS). The validation is performed by comparing SAS parameters calculated using `Cont2SAS` with measured ones. A validated simulation can also be used to retrieve nanostructure from SAS data because a direct retrieval of nanstructure from SAS data is not possible [@billinge2007problem]. Alternatively to simulating continuum nanostructures based on physics-based equations, they can also be simulated based on the user's knowledge of the sample to retrieve nanostructures from SAS data.

![Workflow of SAS pattern calculation: [left] Mesh generation, [middle] SLD assignment, [right] SAS pattern calculation. \label{fig:sas_workflow}](figures/workflow.png)

Both physics- and knowledge-based simulated structures are expected to output either Scattering Length Density (SLD) ($\beta$) values or a set of variables (e.g. local molar density ($\rho_{\text{m}}$) and composition ($\chi$)) from which SLD values can be calculated. `Cont2SAS` creates a mesh, assigns SLD based on the simulated values, and calculates SAS pattern from them, as shown in \autoref{fig:sas_workflow}.

![Workflow of effective cross-section $(\sigma_{\text{eff}})$ calculation: [left] Calculated SAS pattern at different time steps, [right] $\sigma_{\text{eff}}$ calculated from SAS patterns at different time steps: .\label{fig:sig_eff_workflow}](figures/sig_eff.png)

For some materials, the nanostructure does not change over time but the chemical composition does (e.g., ball-milled hydrogen storage materials). For such a scenario, the time evolution of the count rate is a useful parameter [@aslan2019high]. This count rate per incident unit flux is named `effective cross-section` $(\sigma_{\text{eff}})$, and can be calculated using `Cont2SAS`. \autoref{fig:sig_eff_workflow} demonstrates such a calculation from a series of SAS patterns. The calculated $\sigma_{\text{eff}}$ must be multiplied by an empirical factor before comparing with measured neutron count rate.

# Conclusion

`Cont2SAS` provides the much needed software platform for calculating SAS pattern from continuum simulations of nanostructures. The addition of effective cross-section in the software package is going be helpful for analyzing powder-like structures. One can also retrieve continuum nanostructures from SAS data using simulated structures.

# Acknowledgements

This publication was written in the context of the work of the consortium DAPHNE4NFDI in
association with the German National Research Data Infrastructure (NFDI) e.V. NFDI is financed by
the Federal Republic of Germany and the 16 federal states and the consortium is funded by the
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number
460248799. The authors would like to thank for the funding and support. Furthermore, thanks go to
all institutions and actors who are committed to the association and its goals.

# References