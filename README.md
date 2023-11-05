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

Another type of plot, that is also useful for extracting important features is 
__`waffle`__ plot, by default showing any feature that at least has contributed 
0.5% to the overall explained SHAP values across features. 

```r
shapley.plot(result, plot = "waffle")
```

<img src='man/figures/waffle.png' align="center" height="400" />

### Significance testing across features

The bar plot above shows the weighted mean and weighted confidence intervals of SHAP contributions of different features. The confidence intervals already provide information whether the differencce between two features is due to chance. However, we can also perform a significance test to examine whether the difference between two features is statistically significant, using __`shapley.test`__ function. The significance test is based on permutation test, with a sample of 5000 permutations by default. Let's say, we want to examine whether the difference between "GLEASON" and "DPROS" features is due to chance:

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

## Specifying number of top features
Traditionally, specifying number of important feature with highest SHAP values was arbitrary and different scientific articles report, for example, top 10, top 15, top 20, etc. number of features with the hihgest SHAP contributions, yet, without considering the variations between different models. 

However, computing weighted means and weighted 95% confidence intervals provides the possibilities for automated procedures of selecting features that are reliably contributing to the model. For example, the default strategy of the __`shapley`__ package is to select features that their lower weighted 95% confidence interval of relative mean SHAP is above the cutoff value of __`0.01`__. In other words, a feature that has a reliable relative SHAP contribution of 1% or higher (relative to feature with highest contribution) will be selected as important feature. This procedure is applied in the __`bar`__ plot, where the features are sorted based on their weighted mean SHAP values and the cutoff is applied to the lower weighted 95% confidence interval.

This criteria can be changed to "`mean`", where the cutoff is applied for weighted means instead of weighted confidence intervals (THIS IS EXPERIMENTAL! probably it is not a good practice because it ignores the variability between the models, but it is worth further studying). Another criteria is "`shapratio`", which defines a cutoff for minimum weighted mean SHAP value compared to all SHAP values contributed by all features (THIS IS ALSO EXPERIMENTAL). The latter is applied in the __`waffle`__ plot, where selected features at least have 0.5% of contribution of the overall weighted mean SHAP values across all features. 

Of the two procedures "`lowerCI`" and "`shapratio`", both have their own strengths and weaknesses. The "`lowerCI`" concerns the variability between different models and the "`shapratio`" concerns the variability between different features. One might consider implementing both procedures to see to what extent these two procedures are aligned and useful in practice. This is an area that requires active research and testing and future research should particularly address this ideas. 

## SHAP contributions of Stacked Ensemble Models

Stacked ensemble models combine the predictions of multiple models and weight the predictions based on the performance of the base-learner models. A similar strategy is applied by the __`shapley`__ software and thus, the weighted mean SHAP contributions of stacked ensemble models is computed similar to a grid of fine-tuned models as shown above. Whether the specified __`models`__ argument in the __`shapley`__ function is a `h2o` grid or `h2o ensemble`, is considered automatically and there is no need for the use to specify the class of the model object.

## Supported Machine Learning models

The package is designed to work with any machine learning grid or stacked ensemble model that are developed by the [**`h2o`**](https://h2o.ai/blog/2022/shapley-values-a-gentle-introduction/) package 
as well as [**`autoEnsemble`**](https://cran.r-project.org/package=autoEnsemble) 
R package. 



