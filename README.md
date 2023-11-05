> __CITATION__: _Haghish, E. F. (2023). Weighted Mean Shapley Values with Confidence Intervals for Machine Learning Grids and Stacked Ensembles [computer software]. Url: <https://github.com/haghish/shapley>_

<a href="https://github.com/haghish/shapley"><img src='man/figures/logo.png' align="right" height="250" /></a>
  
  `shapley` : Weighted Mean Shapley Values with Confidence Intervals for Machine Learning Grids and Stacked Ensembles
================================================================================

The `shapley` R package provides a simple and efficient method for calculating the weighted mean and weighted confidence interval of Shapley values for machine learning grids and stacked ensembles, resulting in more stable and reliable Shapley values, which also reflect the variability of SHAP values across fine-tuned models. Understanding the SHAP variability 
across fine-tuned models leads to a more transparent, reliable, and potentially 
more reproducible practice of identifying the most important features in a model,
instead of reporting the Shapley values of a single _best_ model. 

The problem with reporting the Shapley values of a single _best_ model is that 
under __severe class imbalance__ (class rarity resulting from low prevalence outcome), there are no global criteria for defining the _best_ model. As a result, the SHAP values 
of a single _best_ model may not be representative of the SHAP values of the other 
models. Under severe class imbalance, the stacked ensemble (meta-lerner) algorithms are more 
likely to produce a reliable model that represents the strength of multiple base-learners 
(component models that are used to train the stacked ensemble meta-learner). However, 
__to date, there has been no method nor software for computing SHAP values for stacked 
ensemble models__. This limitation is addressed with the methodology developed by 
this R package. 

The package is designed to work with any machine learning grid or stacked ensemble model that is supported by the [**h2o**](https://h2o.ai/blog/2022/shapley-values-a-gentle-introduction/) package. 

> To be continued. 

