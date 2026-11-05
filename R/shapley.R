# check sample_size argument

# @param method Character. Method for selecting important features based on WMSHAP.
#               The default is \code{"mean"}; the alternative is \code{"lowerCI"}.

#' @title Weighted Mean SHAP (WMSHAP) and Confidence Interval for Multiple Models
#'
#' @description
#' Computes Weighted Mean SHAP ratios (WMSHAP) and confidence intervals to assess feature
#' importance across a set of near-optimal models. Instead of reporting SHAP
#' contributions for a single model, this function summarizes feature importance
#' across multiple models and weights each model by a chosen performance metric.
#' Models trained by the \code{h2o} machine learning platform use the native H2O procedure.
#' Other models can be explained using \code{treeshap}, \code{kernelshap},
#' \code{shapr}, or the native \code{gpboost} SHAP method.
#'
#' @details
#' If the near-optimal set contains models requiring
#' different non-H2O backends, kernelshap is used for all included models
#' to maintain a common calculation method. The \code{shapr} backend can be
#' selected explicitly to calculate dependence-aware SHAP values; it requires
#' installing the optional \code{shapr} package.
#'
#'    The function works as follows:
#'    \enumerate{
#'      \item For each model, SHAP contributions are computed on \code{newdata}.
#'      \item For each model, feature-level absolute SHAP contributions are aggregated and
#'            converted to a \emph{ratio} (share of total absolute SHAP across features).
#'      \item Models are weighted by a normalized performance metric (e.g., \code{"r2"} for regression or
#'            \code{"auc"} / \code{"aucpr"} for classification).
#'      \item The weighted mean SHAP ratio (WMSHAP) is computed for each feature, along with an
#'            interval across models.
#'    }
#'
#' @param models A model, which can be a fitted R model, a named list of models,
#' an H2O AutoML object, H2O grid object, \code{autoEnsemble} object,
#'               a character vector of H2O model IDs. a wide range of R models are supported.
#' @param newdata A data frame (for R models) or an \code{H2OFrame} (for H2O models),
#' SHAP and WMSHAP contributions are computed on these observations.
#' @param background_data Data frame containing reference observations.
#'   Internally passed as \code{data} to \code{treeshap::unify()},
#'   \code{bg_X} to \code{kernelshap::kernelshap()}, and
#'   \code{x_train} to \code{shapr::explain()}. This argument is not needed
#'   for models trained with 'h2o' or 'gpboost' packages. background_data can be
#'   the training set with predictor columns only.
#' @param pred_fun Optional prediction function of the form
#'   \code{function(model, newdata)}. kernelshap package accepts multiple outputs;
#'   shapr package requires one numeric prediction per observation, selected using
#'   \code{target_class} when necessary. This argument is ignored for models trained
#'   with 'h2o' package, or when the 'treeshap' package is defined as the backend.
#' @param performance_metric Character. Performance metric used to weight models.
#'                           Options are \code{"r2"} (regression), \code{"aucpr"},
#'                           \code{"auc"}, and \code{"f2"}
#'                           (classification metrics).
#' @param cutoff Numeric. WMSHAP mean cutoff used to identify and select important
#'   features. The recommended default is 1 divided by the number of predictors
#'   in the model.
#' @param top_n_features Integer or \code{NULL}. If specified, select the top N features.
#'                       All features are used to compute global importance ratios;
#'                       this argument controls feature selection afterwards.
#' @param sample_size Integer. Number of rows in \code{newdata} used for SHAP assessment.
#'                    Defaults to 1000 rows. Increasing this can slows down WMSHAP computation,
#'                    but improves its accuracy and stability.
#'                    Note that for 'shapr' package, this argument defines of Monte Carlo samples
#'   passed to \code{shapr::explain()}.
#' @param plot Logical. If \code{TRUE}, plots the WMSHAP summary using \code{shapley.plot()}.
#'

#' @param backend Character. Specifies the backend used to compute SHAP values.
#'   the default is "auto", which detects the appropriate backend based on the
#'   models' class (assuming they are trained with a single learner).
#'   Alternative available options are \code{"kernelshap"} (universal), \code{"h2o"} (only for h2o models),
#'   \code{"gpboost"} (only for gpboost models), \code{"treeshap"}
#'   (for supported tree-based models), and
#'   \code{"shapr"}. Each backend uses a different method to estimate SHAP
#'   values. The \code{gpboost} and
#'   \code{shapr} packages are optional dependencies that must be installed
#'   separately. Native GPBoost SHAP values explain only the boosted-tree
#'   component, excluding Gaussian-process and random effects.
#'   Note that these backends compute SHAP values in different ways.
#' @param model_performance Named numeric vector containing one model performance
#'   estimate per non-H2O model. Use comparable cross-validation estimates,
#'   such as AUCPR, for all models. Names should correspond to model list names
#'   (see the examples).
#' @param target_class Name or index of the prediction class to explain for
#'   models returning more than one prediction per observation. The second
#'   class is used by default in binary classification.
#' @param shapr_method Character vector. Method used by
#'   \code{shapr::explain()} to estimate feature contributions.
#'   Available methods are \code{"empirical"}, \code{"gaussian"},
#'   \code{"copula"}, \code{"ctree"}, \code{"arf"},
#'   \code{"vaeac"}, \code{"categorical"}, \code{"independence"},
#'   \code{"timeseries"}, \code{"regression_separate"}, and
#'   \code{"regression_surrogate"}. The default is
#'   \code{"empirical"}.
#' @param n_core Integer. Number of parallel R workers for computing SHAP
#'   contributions across fitted models. The default \code{-1} uses all
#'   detected physical CPU cores, limited to the number of models. Use
#'   \code{1} for sequential computation or a positive integer for a fixed
#'   number of workers. Each worker explains one model at a time. This
#'   argument does not control internal parallelism in the SHAP backends.
#' @param ... arguments that are used internally for compatibility with older
#'            versions
#'
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables head
#' @importFrom stats weighted.mean predict setNames
#' @importFrom h2o h2o.getModel h2o.auc h2o.aucpr h2o.r2
#'             h2o.F2 h2o.shap_summary_plot
#' @importFrom ggplot2 aes coord_flip element_text expansion geom_col
#'             geom_errorbar geom_hline ggplot ggtitle guide_colourbar
#'             guide_legend guides labs margin scale_colour_gradient2
#'             scale_x_discrete scale_y_continuous theme theme_classic xlab ylab
#' @return An object of class \code{"shapley"} (a named list) containing:
#' \describe{
#'   \item{ids}{Character vector of model IDs originally supplied or extracted.}
#'   \item{included_models}{Character vector of included model IDs.}
#'   \item{ignored_models}{Data frame of excluded models and their performance.}
#'   \item{weights}{Legacy rescaled model weights (sum equals the number of models).}
#'   \item{model_weights}{Normalized model-performance weights (sum equals 1).}
#'   \item{results}{Data frame of row-level SHAP contributions across models.}
#'   \item{summaryShaps}{Data frame of global WMSHAP ratios and intervals.}
#'   \item{selectedFeatures}{Character vector of selected important features.}
#'   \item{feature_importance}{List of feature importance values by model.}
#'   \item{contributionPlot}{Plot data for local SHAP contributions.}
#'   \item{plot}{A ggplot object when \code{plot = TRUE}.}
#'   \item{localShaps}{Individual-level WMSHAP and between-model intervals
#'         for all supported models, including H2O.}
#'   \item{model_backends}{Selected SHAP backend for each included non-H2O model.}
#'   \item{explanation_scope}{For native GPBoost SHAP, \code{"gpboost_tree_component"}.
#'         This excludes random effects and Gaussian-process contributions.}
#' }
#' @note SHAP values from different backends are only directly comparable if they
#'   explain the same prediction output and use compatible feature-dependence assumptions.
#'   The intervals summarize between-model variation, not population uncertainty.
#'   The H2O implementation directly retrieves local SHAP contributions using
#'   \code{h2o::h2o.predict_contributions()}, without creating a SHAP plot per model.
#'   Internal performance weights sum to 1; legacy output \code{weights}
#'   sums to the number of included models for compatibility, while
#'   \code{model_weights} contains the normalized weights.
#'   Sampling uncertainty from shapr is different and is not included in these intervals.
#'   For GPBoost mixed-effects models, native SHAP contributions describe the
#'   boosted-tree component only. They do not explain group-level random effects,
#'   Gaussian-process effects, or the entire response prediction. Model weights
#'   may be based on full-model cross-validation performance, but the explained
#'   component remains the boosted-tree function.
#'
#' @examples
#' \dontrun{
#'
#' # ============================================================
#' # An Example Using h2o Package
#' # ============================================================
#' library(shapley)
#' library(h2o)
#' h2o.init(ignore_config = TRUE, nthreads = 2,
#'          bind_to_localhost = FALSE, insecure = TRUE)
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#' y <- "CAPSULE"
#' prostate[, y] <- as.factor(prostate[, y])
#' grid <- h2o.grid(algorithm = "gbm", y = y,
#'                  training_frame = prostate,
#'                  hyper_params = list(ntrees = seq(1, 50)),
#'                  grid_id = "ensemble_grid", nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#' result <- shapley(models = grid, newdata = prostate,
#'                   performance_metric = "aucpr")
#'
#' # ============================================================
#' # An Example Using ranger Package
#' # ============================================================
#' library(tidymodels)
#' library(ranger)
#' library(shapley)
#'
#' # 1. prepare the dataset for training and testing
#' prostate <- read.csv(system.file("extdata", "prostate.csv", package = "h2o"))
#' prostate$ID <- NULL
#' prostate$CAPSULE <- factor(prostate$CAPSULE, levels = c(1, 0))
#'
#' set.seed(2026)
#' split <- initial_split(prostate, prop = 0.80, strata = CAPSULE)
#' train <- training(split)
#' test <- testing(split)
#' predictors <- setdiff(names(train), "CAPSULE")
#'
#' # 2. define a grid for randomforest configurations.
#' rf <- rand_forest(
#'   trees = tune(),
#'   mtry = tune(),
#'   min_n = tune()
#' ) |>
#'   set_engine("ranger", num.threads = 1L, seed = 2026L) |>
#'   set_mode("classification")
#'
#' wf <- workflow() |>
#'   add_model(rf) |>
#'   add_formula(CAPSULE ~ .)
#'
#' grid <- tidyr::crossing(
#'   trees = c(100L, 200L, 300L, 400L, 500L),
#'   mtry = c(2L, 3L, 4L, 5L, 6L, 7L),
#'   min_n = c(2L, 4L, 6L, 8L)
#' )
#'
#' # 3. evaluate the configurations using 10-fold CV and AUCPR as evaluation metrics.
#' folds <- vfold_cv(train, v = 10, strata = CAPSULE)
#'
#' cv <- tune_grid(
#'   wf,
#'   resamples = folds,
#'   grid = grid,
#'   metrics = metric_set(pr_auc),
#'   control = control_grid(verbose = TRUE, save_pred = FALSE,
#'                          save_workflow = FALSE, event_level = "first")
#' )
#'
#' # mean AUCPR across CV folds
#' scores <- collect_metrics(cv) |>
#'   dplyr::filter(.metric == "pr_auc", is.finite(mean)) |>
#'   dplyr::arrange(dplyr::desc(mean))
#'
#' # 4: select the near-optimal configurations. for example,
#' # retain top models with at least 98% of the best mean AUCPR.
#' near <- scores |>
#'   dplyr::filter(mean >= 0.98 * max(scores$mean))
#'
#' # 5: fit each retained configuration on the complete training data.
#' models <- lapply(seq_len(nrow(near)), function(i) {
#'   pars <- near[i, c("trees", "mtry", "min_n")]
#'   fitted <- finalize_workflow(wf, pars) |>
#'     fit(data = train)
#'   workflows::extract_fit_engine(fitted)
#' })
#'
#' names(models) <- paste0("model", seq_along(models))
#' cv_aucpr <- stats::setNames(near$mean, names(models))
#'
#' # computing WMSHAP using treeshap package
#' # ------------------------------------------------------------
#' result_tree <- shapley(
#'   models = models,
#'   newdata = test[, predictors],
#'   background_data = train[, predictors],
#'   model_performance = cv_aucpr,
#'   performance_metric = "aucpr",
#'   backend = "treeshap",
#'   target_class = "1",
#'   plot = TRUE
#' )
#'
#' # computing WMSHAP using kernelshap package
#' # ------------------------------------------------------------
#' pred_fun <- function(model, X) {
#'   requireNamespace("ranger", quietly = TRUE)
#'   as.numeric(stats::predict(model, data = as.data.frame(X), num.threads = 1)$predictions[, "1"])
#' }
#'
#' result_kernel <- shapley(
#'   models = models,
#'   newdata = test[, predictors],
#'   background_data = train[, predictors],
#'   model_performance = cv_aucpr,
#'   performance_metric = "aucpr",
#'   backend = "kernelshap",
#'   pred_fun = pred_fun,
#'   target_class = "1",
#'   n_core = 4,
#'   plot = FALSE
#' )
#'
#' # omputing WMSHAP using shapr package
#' # ------------------------------------------------------------
#' result_shapr <- shapley(
#'   models = models,
#'   newdata = test[, predictors],
#'   background_data = train[, predictors],
#'   model_performance = cv_aucpr,
#'   performance_metric = "aucpr",
#'   backend = "shapr",
#'   pred_fun = pred_fun,
#'   target_class = "1",
#'   shapr_method = "empirical",
#'   sample_size = 1000,
#'   plot = TRUE
#' )
#'
#' # WMSHAP global results (feature-level)
#' # ------------------------------------------------------------
#' print(result_tree$summaryShaps)
#' print(result_kernel$summaryShaps)
#' print(result_shapr$summaryShaps)
#'
#' # WMSHAP local results (individual-level)
#' # ------------------------------------------------------------
#' print(head(result_tree$localShaps))
#' print(head(result_kernel$localShaps))
#' print(head(result_shapr$localShaps))
#'
#' }
#' @export
#' @author E. F. Haghish

shapley <- function(models,
                    newdata,
                    background_data = NULL,
                    pred_fun = NULL,
                    model_performance = NULL,
                    performance_metric = "r2",
                    backend = "auto",


                    sample_size = 1000L,
                    n_core = -1L,
                    cutoff = 0,
                    plot = TRUE,

                    top_n_features = NULL,
                    target_class = NULL,

                    #shapr arguments
                    shapr_method = "empirical",
                    ...
                    ) {

  # check the ... arguments
  # ============================================================
  hidden_args <- c("minimum_performance", "n_models", "shapr_phi0", "method")
  stopifnot("incompatible '...' arguments" = (names(list(...)) %in% hidden_args))
  minimum_performance    <- threeDots(name = "minimum_performance", ..., default = 0)
  n_models               <- threeDots(name = "n_models", ..., default = 2)
  shapr_phi0             <- threeDots(name = "shapr_phi0", ..., default = NULL)
  method                 <- threeDots(name = "method", ..., default = "mean")

  # if backend is shapr, change the default of sample_size
  if (backend == "shapr") {
    if (is.null(sample_size)) {
      message("shapr requires the 'sample_size' argument, which is set to NULL")
      sample_size <- 1000L
    }
  }

  # WMSHAP requires at least two fitted models.
  # ============================================================
  if (length(n_models) != 1L || !is.numeric(n_models) ||
      !is.finite(n_models) || n_models < 2 ||
      n_models != as.integer(n_models)) {
    stop("'n_models' must be an integer greater than or equal to 2 for WMSHAP")
  }
  n_models <- as.integer(n_models)

  # STEP 0: Determine the number of model-level workers.
  # ============================================================
  n_core <- shapley.cores(n_core)

  # STEP 1: Identify the SHAP backend.
  # ============================================================
  backend <- match.arg(backend, c("auto", "h2o", "gpboost", "treeshap", "kernelshap", "shapr"))
  model_info <- shapley.models(models)

  # STEP 1: Keep the original H2O computation unchanged.
  # ============================================================
  if (model_info$h2o) {
    if (!requireNamespace("h2o", quietly = TRUE)) {
      stop("Package 'h2o' is required for H2O models. Install it with install.packages('h2o').")
    }
    if (!backend %in% c("auto", "h2o")) {
      stop("H2O models require backend = 'h2o' or 'auto'")
    }
    if (!is.null(model_performance)) {
      stop("H2O already retrieves performance estimates. Leave 'model_performance' as NULL.")
    }
    return(shapley.h2o(
      models = models, newdata = newdata, plot = plot,
      performance_metric = performance_metric,
      performance_type = "xval",
      minimum_performance = minimum_performance,
      method = method, cutoff = cutoff,
      top_n_features = top_n_features,
      n_models = n_models,
      sample_size = sample_size,
      n_core = n_core
    ))
  }

  # STEP 2: Use native GPBoost SHAP, TreeSHAP, Kernel SHAP, or shapr.
  # ============================================================
  if (backend == "h2o") stop("Only H2O models use the H2O backend")
  return(shapley.external(
    models = models, newdata = newdata, plot = plot,
    performance_metric = performance_metric,
    minimum_performance = minimum_performance,
    method = method, cutoff = cutoff,
    top_n_features = top_n_features,
    n_models = n_models, sample_size = sample_size,
    shap_backend = backend,
    background_data = background_data,
    pred_fun = pred_fun,
    model_performance = model_performance,
    target_class = target_class,
    shapr_approach = shapr_method,
    shapr_phi0 = shapr_phi0,
    shapr_n_MC_samples = sample_size,
    n_core = n_core
  ))
}
