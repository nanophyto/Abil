---
title: 'Abil: A python package for the interpolation of aquatic biogeochemical datasets'
tags:
  - Python
  - biogeochemistry
  - ocean
  - machine learning
  - species distribution modelling
  - plankton
  - random forests
  - XGBoost
  - Bagged Nearest Neighbors
  - ensemble-based machine learning
  - zero-inflated regression
  - area of applicability
authors:
  - name: Joost de Vries
    orcid: 
    affiliation: 1
  - name: Nicola A. Wiseman
    orcid: 0000-0001-9296-7566
    affiliation: 1
  - name: Levi John Wolf
    orcid: 
    affiliation: 1
affiliations:
 - name: School of Geographical Sciences, University of Bristol, BS8 1HB, UK
   index: 1
date: 5 February 2025
bibliography: paper.bib
---

# Summary

Our oceans play a critical role in regulating the Earth's climate and sustaining local economies through fisheries and tourism [@moreno:2009; @dyck:2010]. However, the vast size of the ocean means that observations are inherently sparse, posing significant challenges to contextualizing these observations on a global scale [@hauck:2023]. Ensemble-based machine learning approaches offer an exciting avenue to address this challenge. However, the complexity of implementing these algorithms, combined with the need for extensive pre-processing and post-processing, highlights the necessity for efficient, reproducible numerical tools. Here we provide a Python package for training, predicting, and post-processing a machine learning ensemble to facilitate the global interpolation of sparse observational datasets, such as those from oceanographic cruises.

# Statement of Need

`Abil` is a Python package to interpolate sparse observations using ensemble-based machine learning algorithms. Oceanographic data is sparse in terms of spatial and temporal distribution due to the nature of collection during oceanographic cruises, and requires a more informed approach than traditional gap-filling interpolation. Previous studies have utilized machine-learning methods as a solution to this problem, but often underlying code is difficult to access and deploy. To improve reproducibility within the community, `Abil` was developed as an open-source, user-friendly interface through which machine learning methods can be more easily be implemented.

The API for `Abil` was designed to provide a user-friendly interface to fast implementations of `scikit-learn` and `XGBoost` ensemble-based machine learning algorithms. In addition, it includes essential tools such as pre-implemented pipelines, zero-inflated regressor support, and comprehensive post-processing functionalities such as latitudinally weighted integration and area of applicability estimates. Beyond scikit-learn, `Abil` leverages libraries like `Xarray`, `Pandas`, and `Numpy` for efficient data manipulation, as well as `scikit-bio` for biodiversity metrics.

The package is optimized for parallel processing, and provides vignettes of high performance use. Abil.py is thus particularly suited to modeling the distribution of species, genes, and transcripts, as well as biogeochemical processes such as calcite production. Unlike many existing tools focused on predicting species occurrence (e.g., `elapid` and `biomod2`), `Abil` specializes in regression challenges, enabling the prediction of abundances and rates. This focus on regression complements existing packages and fills a critical gap in the application of statistical models to ocean ecology and biogeochemistry.

By combining a user-friendly interface, parallel processing capabilities, and a specific focus on regression problems, `Abil` facilitates novel scientific explorations of sparse oceanic datasets. Its versatility and computational efficiency enable researchers to address complex challenges in ocean biogeochemistry and ecology with greater ease and accuracy.

# Documentation

`Abil` documentation can be found through [github](https://nanophyto.github.io/Abil). The documentation includes instruction for installing the model locally and running on an HPC system. The model process consists of tuning the model (class: tune), predicting the model (class: predict), and postprocessing the model (class: post). The documentation includes useage examples for running these model steps.

# Acknowledgements

This work was supported by funding from the UK Research and Innovation Natural Environment Research Council (CoccoTrait, NE/X001261/1).

# References

