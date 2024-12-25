---
title: 'Abil: A python package for the interpolation of aquatic biogeochemical datasets'
tags:
  - Python
  - biogeochemistry
  - ocean
  - machine learning
  - species distribution modelling
authors:
  - name: Joost de Vries
    orcid: 
    affiliation: 1
  - name: Nicola Wiseman
    orcid: 
    affiliation: 1
  - name: Levi John Wolf
    orcid: 
    affiliation: 1
affiliations:
 - name: School of Geographical Sciences, University of Bristol, BS8 1HB, UK
   index: 1
date: 13 August 2017
bibliography: paper.bib
---

# Summary

Our oceans play a critical role in regulating the Earth's climate and sustaining local economies through fisheries and tourism. However, the vast size of the ocean means that observations are inherently sparse, posing significant challenges to contextualizing these observations on a global scale. Ensemble-based machine learning approaches offer an exciting avenue to address this challenge. However, the complexity of implementing these algorithms, combined with the need for extensive pre-processing and post-processing, highlights the necessity for efficient, reproducible numerical tools.

# Statement of Need

`Abil` is a Python package developed by ocean biogeochemists to interpolate sparse observations using ensemble-based machine learning algorithms. Python's robust ecosystem provides a high-level user interface that is computationally efficient enough to perform extensive model optimization, making it an ideal platform for this tool.

The API for `Abil` was designed to provide a user-friendly interface to fast implementations of scikit-learn ensemble-based machine learning algorithms. In addition, it includes essential tools such as pre-implemented pipelines, 2-phase models, and comprehensive post-processing functionalities. Beyond scikit-learn, `Abil` leverages libraries like `Xarray`, `Pandas`, and `Numpy` for efficient data manipulation, as well as `scikit-bio` for biodiversity metrics.

The package is optimized for parallel processing, making it particularly suited to modeling the distribution of species, genes, and transcripts, as well as biogeochemical processes such as photosynthesis and calcium carbonate fixation. Unlike many existing tools focused on predicting species occurrence (e.g., `elapid` and `biomod2`), `Abil` specializes in regression challenges, enabling the prediction of abundances and rates. This focus on regression complements existing packages and fills a critical gap in the application of statistical models to ocean ecology and biogeochemistry.

By combining a user-friendly interface, parallel processing capabilities, and a specific focus on regression problems, `Abil` facilitates novel scientific explorations of sparse oceanic datasets. Its versatility and computational efficiency enable researchers to address complex challenges in ocean biogeochemistry and ecology with greater ease and accuracy.

# Acknowledgements

This work was supported by funding from the Natural Environment Research Council (NERC).

# References

