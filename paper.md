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

Our oceans play a critical role in regulating the Earth's climate and sustaining local economies through fisheries and tourism. However, the vast size of the ocean means that observations are inherently sparse, posing significant challenges to contextualizing these observations on a global scale. Ensemble-based machine learning approaches offer an exciting avenue to address this challenge. However, the complexity of implementing these algorithms, combined with the need for extensive pre-processing and post-processing, highlights the necessity for efficient, reproducible numerical tools. Here we provide a Python package for training, predicting, and post-processing a machine learning ensemble to facilitate the global interpolation of sparse observational datasets, such as those from oceanographic cruises.

# Statement of Need

`Abil` is a Python package to interpolate sparse observations using ensemble-based machine learning algorithms. Oceanographic data is sparse in terms of spatial and temporal distribution due to the nature of collection during oceanographic cruises, and requires a more informed approach than traditional gap-filling interpolation. Previous studies have utilized machine-learning methods as a solution to this problem, but often underlying code is difficult to access and deploy. To improve reproducibility within the community, `Abil` was developed as an open-source, user-friendly interface through which machine learning methods can be more easily be implemented.

The API for `Abil` was designed to provide a user-friendly interface to fast implementations of `scikit-learn` and `XGBoost` ensemble-based machine learning algorithms. In addition, it includes essential tools such as pre-implemented pipelines, zero-inflated regressor support, and comprehensive post-processing functionalities such as latitudinally weighted integration and area of applicability estimates. Beyond scikit-learn, `Abil` leverages libraries like `Xarray`, `Pandas`, and `Numpy` for efficient data manipulation, as well as `scikit-bio` for biodiversity metrics.

The package is optimized for parallel processing, and provides vignettes of high performance use. Abil.py is thus particularly suited to modeling the distribution of species, genes, and transcripts, as well as biogeochemical processes such as calcite production. Unlike many existing tools focused on predicting species occurrence (e.g., `elapid` and `biomod2`), `Abil` specializes in regression challenges, enabling the prediction of abundances and rates. This focus on regression complements existing packages and fills a critical gap in the application of statistical models to ocean ecology and biogeochemistry.

By combining a user-friendly interface, parallel processing capabilities, and a specific focus on regression problems, `Abil` facilitates novel scientific explorations of sparse oceanic datasets. Its versatility and computational efficiency enable researchers to address complex challenges in ocean biogeochemistry and ecology with greater ease and accuracy.

# Usage Example

There are three steps to running `Abil`, outlined below. Additional documentation for `Abil` is available through [github](https://nanophyto.github.io/Abil). 

## Tune

To initialize the model, the training dataset `d` (`pandas.DataFrame`) is subset to `X_train` and `y`, where `y` is the target data, and `X_train` is the associated environmental predictors. These are then passed in to the tuning class using m = tune(X_train, y, model_config). The training is then performed by calling m.train(model=model, regressor=True), where model is one of four supported models ("xgb", "knn", "rf", "gp"). The output will print "execution time: XXX seconds" upon completion.

## Predict

To run the model prediction, the training dataset `d` is again subset to `X_train` and `y`. Additionally, the fully resolved environmental data "X_predict" is read in as a `pandas.DataFrame`. These are passed to the prediction class using m = predict(X_train, y, model_config) and then called by m.make_prediction(). The output will print "execution time: XXX seconds" upon completion.

## Post

To post-process the model, initialize the class with m = post(model_config). Additional postprocessing includes: m.merge_performance(model=model), which summarizes the performance statistics for each target (i.e. MAE, rMSE, R2) and exports them as a .csv; m.merge_parameters(model=model), which summarizes the optimized hyperparameters and exports them as a .csv; m.merge_env(X_predict), which merges the model prediction with the fully resolved environmental data and exports them as a NetCDF file; m.export_ds(file_name)/m.export_csv(file_name), which exports the processed dataset at a NetCDF file or .csv respectively; as well as other post-processing functions such as those needed for calculating global integrals of the processed dataset. 

# Acknowledgements

This work was supported by funding from the Natural Environment Research Council (NERC).

# References

