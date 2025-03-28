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

Abil is a Python package to interpolate sparse observations using ensemble-based machine learning algorithms. Oceanographic data is sparse in terms of spatial and temporal distribution due to the nature of collection during oceanographic cruises and requires a more informed approach than traditional gap-filling interpolation. Previous studies have utilized machine-learning methods as a solution to this problem, but often underlying code is difficult to access and deploy. To improve reproducibility within the community, Abil was developed as an open-source, user-friendly interface through which machine learning methods can be more easily implemented. 

The API for Abil was designed to provide a user-friendly interface to fast implementations of scikit-learn [@ref] and XGBoost [@ref] ensemble-based machine learning algorithms. The user interface centers around three Python classes (optimization, prediction and post-processing) and three ensemble-based machine learning algorithms: random forests, bagged KNN and bagged XGBoost. Abil uses a user-defined YAML [@ref] model configuration for model setup, which contains model specifications such as the model features to include, hyper-parameter values, and the number of cross-folds to be used. By containing all model specifications inside a single and easy to read YAML, each model run is highly traceable and reproducible.   

In addition, Abil includes essential tools such as pre-implemented pipelines which include environmental feature scaling – a step which is required for algorithms such as nearest neighbor algorithms, optional predictor log transformation – a step desirable in cases where high-value outliers can skew predictions, and zero-stratified cross validation for predictors where absences are more common than occurrences (`zero-inflation`, which is common for ocean biogeochemical observations). For zero-inflated models we also provide zero-inflated regressor support, through the implementation of a 2-phase model pipeline which includes a classifier step to predict presence/absence before a regressor is applied in samples where presence is inferred.  

To estimate model prediction uncertainty, Abil leverages predictions of the ensemble-based machine learning members (i.e. decision trees for random forests [@ref], and bags for bagged-KNN and bagged-XGBoost). To infer prediction quantiles, predictions are made for each ensemble member, which are then combined to estimate the 95th percentiles using loss-weighted quantiles. To reduce RAM requirements, this step is implemented using chunking, and `loky` multiprocessing [@ref]. 

To aid in model analysis Abil supports comprehensive post-processing functionalities such as latitudinally weighted integration and area of applicability estimates. Beyond scikit-learn, Abil leverages libraries like Xarray [@ref], Pandas [@ref], and Numpy [@ref] for efficient data manipulation, as well as scikit-bio [@ref] for biodiversity metrics. 

The package is optimized for parallel processing through the use of loky multiprocessing [@ref] and provides vignettes of high-performance computing scripts such that it can be easily ported to large scale parallel programming contexts. Abil.py is thus particularly suited to modeling the distribution of species, genes, and transcripts, as well as biogeochemical processes such as organic carbon and calcite production. Unlike many existing tools focused on predicting species occurrence (e.g., elapid and biomod2), Abil specializes in regression challenges, enabling the prediction of abundances and rates. This focus on regression complements existing packages and fills a critical gap in the application of statistical models to ocean ecology and biogeochemistry. 

By combining a user-friendly interface, parallel processing capabilities, and a specific focus on regression problems, Abil facilitates novel scientific explorations of sparse oceanic datasets. Its versatility and computational efficiency enable researchers to address complex challenges in ocean biogeochemistry and ecology with greater ease and accuracy. 

# Documentation

`Abil` documentation can be found through [github](https://nanophyto.github.io/Abil). The documentation includes instruction for installing the model locally and running on an HPC system. The model process consists of tuning the model (class: tune), predicting the model (class: predict), and postprocessing the model (class: post). The documentation includes usage examples for running these model steps.

# Acknowledgements

This work was supported by funding from the UK Research and Innovation Natural Environment Research Council (CoccoTrait, NE/X001261/1).

# References

