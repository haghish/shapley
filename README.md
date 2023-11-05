> __CITATION__: _Haghish, E. F. (2023). Weighted Mean Shapley Values with Confidence Intervals for Machine Learning Grids and Stacked Ensembles [computer software]. Url: <https://github.com/haghish/shapley>_

- - -


<a href="https://github.com/haghish/shapley"><img src='man/figures/logo.png' align="right" height="200" /></a>
  
  __`shapley`__ : Weighted Mean Shapley Values with Confidence Intervals for Machine Learning Grids and Stacked Ensembles
================================================================================

## Introduction

The `shapley` R package provides a simple and efficient method for calculating the weighted mean and weighted confidence interval of Shapley values for machine learning grids and stacked ensembles, resulting in more stable and reliable Shapley values, which also reflect the variability of SHAP values across fine-tuned models. Understanding the SHAP variability across fine-tuned models leads to a more transparent, reliable, and potentially more reproducible practice of identifying the most important features in a model, instead of reporting the Shapley values of a single _best_ model. The problem with reporting the Shapley values of a single _best_ model is that under __severe class imbalance__ (class rarity resulting from low prevalence outcome), there are no global criteria for defining the _best_ model. As a result, the SHAP values of a single _best_ model may not be representative of the SHAP values of the other models. Under severe class imbalance, the stacked ensemble (meta-lerner) algorithms are more likely to produce a reliable model that represents the strength of multiple base-learners (component models that are used to train the stacked ensemble meta-learner). However, __to date, there has been no method nor software for computing SHAP values for stacked ensemble models__. This limitation is addressed with the methodology developed by this R package. In particular, the 
`shapley` software addresses the following shortcomings often found in recent literature of applied machine learning:

### Current limitations in the machine learning literature
> 1. Reporting unstable SHAP values from a single model, without considering the performance of the model nor the cross-variability of SHAP values across fine-tuned 
models
2. Arbitrary number used for selecting "top features", which usually includes a 
predefined number of features, such as top 10, top 20, etc. 
3. Lack of method for computing SHAP values for stacked ensemble models or shap 
contributions of features across all models of a tuned grid search
4. Lack of method for computing confidence intervals for SHAP values, which could 
also be used for significance testing across different features, i.e., to examine 
whether one feature is significantly more important than another feature. 

### Solutions implemented in the `shapley` R package

The `shapley` R package provides a simple and efficient method for calculating the weighted mean and weighted confidence interval of Shapley values for multiple machine learning models. This package takes the performance metrics as a weight to consider SHAP calues computed from different models. Similarly, it computes the weighted 95% confidence intervals, which reflect on the variability of SHAP values across models. This procedure solves the problem resulting from reporting SHAP values from a _single best model_, without considering how the performance of the model reflects the SHAP values. The latter becomes particularly problematic in circumstances where defining "the best model" becomes unreliable or situational, such as under "severe class imbalance proble" (class rarity, resulting from low-prevalence outcome of interest). The methodology implemented in this software provides the possibility for more reliable SHAP contributions across models. This method also provide the ground for significance testing between the features, i.e., assessing whether the difference between SHAP values of two features is not due to random chance.

This software also suggest several possibility for automatic and transparent procedures of specifying "important features", without requiring prespecified number of top features. These methods take different metrics into account to define "what is an important feature", which then, accordingly, number of important features based on SHAP contributions can be selected, eliminating the need for specifying the number of imoortant features in advance (for detail, see below). 

## Examples

To demonstrate how __`shapley`__ can compute SHAP values across a machine learning grid, let's carry out a grid search to fine-tune Gradient Boosting Machines (GBM) algorithm for a binary classification. Next, I will use the grid to compute SHAP contributions across all models and report their weighted mean and weighted 95% confidence intervals. 

```r
library(h2o)            #shapley supports h2o models
library(shapley)

# initiate the h2o server
h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)

# upload data to h2o cloud
prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
prostate <- h2o.importFile(path = prostate_path, header = TRUE)

# run AutoML to tune various models (GBM) for 60 seconds
y <- "CAPSULE"
prostate[,y] <- as.factor(prostate[,y])  #convert to factor for classification

set.seed(10)

#######################################################
### PREPARE H2O Grid (takes a couple of minutes)
#######################################################
# make sure equal number of "nfolds" is specified for different grids
grid <- h2o.grid(algorithm = "gbm", y = y, training_frame = prostate,
                 hyper_params = list(ntrees = seq(1,50,1)),
                 grid_id = "ensemble_grid",

                 # this setting ensures the models are comparable for building a meta learner
                 seed = 2023, fold_assignment = "Modulo", nfolds = 10,
                 keep_cross_validation_predictions = TRUE)

result <- shapley(grid, newdata = prostate, plot = TRUE)
```

In the example above, the `result` object would be a _list of class `shapley`_, which in cludes the information such as weighted mean and weighted confidence intervals as well as other metrics regarding SHAP contributions of different features. 

### Plotting SHAP values

You can use the __`shapley.plot`__ function to plot the SHAP contributions:

1. To plot weighted mean SHAP contributions as well as weighted 95% confidence intervals, pass the `shapley object`, in this example, named `result`, and specify `"bar"`, to create a bar plot:

```r
shapley.plot(result, plot = "bar")
```

<img src='man/figures/bar.png' align="center" height="400" />

Another type of plot, that is also useful for identifying important features is 
__`waffle`__ plot, by default showing any feature that at least has contributed 
0.5% to the overall explained SHAP values across features. 

```r
shapley.plot(result, plot = "waffle")
```

<img src='man/figures/waffle.png' align="center" height="400" />

### Significance testing across features

The bar plot displays the weighted average and confidence intervals for the SHAP contributions of various features, implying whether differences between two features might be due to random variation. To further investigate the statistical significance of these differences, the `shapley.test` function can be employed. This function uses a permutation test, typically with a default of 5000 permutations, to assess significance. For instance, to determine if the observed difference in SHAP contributions between the "GLEASON" and "DPROS" features is not just a random occurrence, we would use this function.

```r
shapley.test(result, features = c("GLEASON", "DPROS"), n = 5000)
```
```
The difference between the two features is significant:
observed weighted mean Shapley difference = 150.014579392817 and p-value = 0
$mean_shapley_diff
[1] 150.0146

$p_value
[1] 0
```

However, if we check the difference between "PSA" and "ID" features, the difference is insignificant, although "PSA" has a slightly higher weighted mean SHAP contribution:

```r
shapley.test(result, features = c("PSA", "ID"), n = 5000)
```
```
The difference between the two features is not significant:
observed weighted mean Shapley difference =2.44791199071206 and p-value = 0.678
$mean_shapley_diff
[1] 2.447912

$p_value
[1] 0.678
```

> Note: the weighted confidence intervals showed in the bar plot do not apply any permutation test.

## Specifying number of top features

Traditionally, the selection of a set number of significant features based on the highest SHAP values varied across scientific publications, with some reporting the top 10, 15, or 20. This selection did not account for the variability between models and often was either an arbitrary number. Other papers reported feature importance of all features, which is not practical for large datasets, and also, doesn't consider if features with neglegible SHAP contributions are statistically significant, given the variability across models. However, the calculation of weighted means and 95% confidence intervals allows for a systematic approach to identify features that consistently contribute to the model, across different models. For instance, the default method in the `shapley` package considers features important if their lower bound of the weighted 95% confidence interval ("`lowerCI`") for relative mean SHAP value exceeds `0.01`. This means any feature with a stable relative contribution of at least 1% - relative to the feature with the highest SHAP - is deemed important. This method is also utilized in the `bar` plot, where features are ranked by their weighted mean SHAP values, and the cutoff is applied to the lower confidence interval. This threshold can be adjusted to "`mean`", which sets the cutoff for weighted means, disregarding model variability—although this is experimental and might not be best practice, because it overlooks the variablitity across models and only considers the weighted means. Another experimental criterion - that seems more plausible than the "`mean`" strategy is "`shapratio`", setting a cutoff for the minimum weighted mean SHAP value as a percentage of the total SHAP contributions from all features. This is demonstrated in the `waffle` plot, where features must contribute at least 0.5% to the overall weighted mean SHAP values to be selected.

Between the "`lowerCI`" and "`shapratio`" methods, each has merits and limitations. "`LowerCI`" addresses variability across models, while "`shapratio`" focuses on feature contribution variability. Implementing both could provide insight into their efficacy and alignment in practice. Future research should explore these methodologies further. 



## SHAP contributions of Stacked Ensemble Models

Stacked ensemble models integrate multiple base learner models' predictions, assigning weights according to each base model's performance. The methodology implemented in `shapley` software employs a similar approach, calculating the weighted mean SHAP contributions for stacked ensemble models just as it would for a fine-tuned grid of models. The `shapley` function's `models` parameter automatically detects whether the input is an `h2o` grid or an `h2o` or `autoEnsemble` stacked ensemble, eliminating the need for users to identify the model object class. 

## Supported Machine Learning models

The package is compatible with machine learning grids or stacked ensemble models created using the [`h2o`](https://h2o.ai/blog/2022/shapley-values-a-gentle-introduction/) package, as well as the [`autoEnsemble`](https://cran.r-project.org/package=autoEnsemble) package in R.
