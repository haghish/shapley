# check sample_size argument

#' @title Weighted Mean SHAP (WMSHAP) and Confidence Interval for Multiple Models
#'
#' @description
#' Computes Weighted Mean SHAP ratios (WMSHAP) and confidence intervals to assess feature
#' importance across a collection of models (e.g., an H2O grid/AutoML leaderboard or
#' base-learners of an ensemble). Instead of reporting SHAP contributions for a single model,
#' this function summarizes feature importance across multiple models and weights each model
#' by a chosen performance metric.
#' Models trained by the \code{h2o} machine learning platform use the native H2O procedure.
#' Other models can be explained using \code{treeshap}, \code{kernelshap},
#' \code{shapr}, or the native \code{gpboost} SHAP method.
#' H2O, TreeSHAP, and Kernel SHAP are installed with SHAPLEY. GPBoost and
#' shapr are optional packages, required only when their methods are used.
#'
#' @details
#' When \code{shap_backend = "auto"}, SHAPLEY uses the H2O backend for H2O
#' models, native GPBoost tree contributions for \code{gpb.Booster} models
#' (requires the optional \code{gpboost} package),
#' TreeSHAP for compatible tree models, and Kernel SHAP for models
#' not supported by TreeSHAP. If the near-optimal set contains models requiring
#' different non-H2O backends, Kernel SHAP is used for all included models
#' to maintain a common calculation method. The \code{shapr} backend can be
#' selected explicitly to calculate dependence-aware SHAP values; it requires
#' installing the optional \code{shapr} package.
#'
#'    The function works as follows:
#'    \enumerate{
#'      \item For each model, SHAP contributions are computed on \code{newdata}.
#'      \item For each model, feature-level absolute SHAP contributions are aggregated and
#'            converted to a \emph{ratio} (share of total absolute SHAP across features).
#'      \item Models are weighted by a performance metric (e.g., \code{"r2"} for regression or
#'            \code{"auc"} / \code{"aucpr"} for classification).
#'      \item The weighted mean SHAP ratio (WMSHAP) is computed for each feature, along with an
#'            interval across models.
#'    }
#'
#' @param models An H2O AutoML object, H2O grid object, \code{autoEnsemble} object,
#'               a character vector of H2O model IDs, a fitted R model, or a named list of models.
#' @param newdata An \code{H2OFrame} for H2O models, or a data frame containing predictors
#'                for other models. SHAP contributions are computed on these observations.
#' @param performance_metric Character. Performance metric used to weight models.
#'                           Options are \code{"r2"} (regression), \code{"aucpr"}, \code{"auc"}, and \code{"f2"}
#'                           (classification metrics).
#' @param standardize_performance_metric Logical. Retained for compatibility.
#'   Both H2O and other models now use the same normalized performance weights,
#'   regardless of this argument. The default is \code{FALSE}.
#' @param performance_type Character. Specify which performance estimate to use:
#'                         \code{"train"} for training data, \code{"valid"}
#'                        for validation, or \code{"xval"} for cross-validation (default).
#'                        Applies to H2O models; for other models supply \code{model_performance}.
#' @param minimum_performance Numeric. Minimum performance metric
#'                            for a model to be included in calculating WMSHAP.
#'                            Models below this threshold are excluded. The default is \code{0}.
#' @param method Character. Method for selecting important features based on WMSHAP.
#'               The default is \code{"mean"}; the alternative is \code{"lowerCI"}.
#' @param cutoff Numeric. Cutoff applied by \code{method} for selecting important features.
#' @param top_n_features Integer or \code{NULL}. If specified, select the top N features.
#'                       All features are used to compute global importance ratios;
#'                       this argument controls feature selection afterwards.
#' @param n_models Integer. Minimum number of included models for WMSHAP.
#'   Must be at least \code{2}; the default is \code{10}. Models that do not
#'   meet \code{minimum_performance} are excluded from this count.
#' @param sample_size Integer. Number of rows in \code{newdata} used for SHAP assessment.
#'                    Defaults to all rows. Reducing this can speed up development runs.
#' @param plot Logical. If \code{TRUE}, plots the WMSHAP summary using \code{shapley.plot()}.
#'
#' @param shap_backend Character. The default \code{"auto"} selects H2O for
#'   H2O models, native \code{gpboost} SHAP for \code{gpb.Booster} models,
#'   \code{treeshap} for compatible tree models, and
#'   \code{kernelshap} otherwise. If different non-H2O model classes would
#'   require different methods, \code{kernelshap} is used for all models.
#'   Specify \code{"h2o"}, \code{"gpboost"}, \code{"treeshap"},
#'   \code{"kernelshap"}, or \code{"shapr"} to select one method.
#'   The GPBoost and shapr packages are installed separately when needed.
#'   Native GPBoost SHAP explains
#'   only the boosted-tree component, excluding Gaussian-process and random effects.
#' @param background_data Data frame containing reference observations.
#'   Internally passed as \code{data} to \code{treeshap::unify()},
#'   \code{bg_X} to \code{kernelshap::kernelshap()}, and
#'   \code{x_train} to \code{shapr::explain()}.
#'   The native GPBoost backend does not use \code{background_data}.
#'   A common representative training subset is recommended for multiple models.
#' @param pred_fun Optional prediction function of the form
#'   \code{function(model, newdata)}. Kernel SHAP accepts multiple outputs;
#'   shapr requires one numeric prediction per observation, selected using
#'   \code{target_class} when necessary.
#' @param model_performance Named numeric vector containing one model performance
#'   estimate per non-H2O model. Use comparable cross-validation estimates,
#'   such as AUCPR, for all models. Names should correspond to model list names.
#' @param target_class Name or index of the prediction class to explain for
#'   models returning more than one prediction per observation. The second
#'   class is used by default in binary classification.
#' @param shapr_approach Character. Feature-dependence approach used by
#'   \code{shapr::explain()}. The default is \code{"empirical"}. Specify
#'   \code{"independence"} when a marginal explanation is desired.
#'   Other approaches supported by shapr can also be specified.
#' @param shapr_phi0 Optional numeric baseline prediction supplied as
#'   \code{phi0} to \code{shapr::explain()}. By default, it is the mean of
#'   model predictions on \code{background_data}. May be a named vector
#'   with one baseline per model.
#' @param shapr_n_MC_samples Positive integer. Number of Monte Carlo samples
#'   passed to \code{shapr::explain()}; the default is 1000.
#' @param n_core Integer. Number of parallel R workers for computing SHAP
#'   contributions across fitted models. The default \code{-1} uses all
#'   detected physical CPU cores, limited to the number of models. Use
#'   \code{1} for sequential computation or a positive integer for a fixed
#'   number of workers. Each worker explains one model at a time. This
#'   argument does not control internal parallelism in the SHAP backends.
#'
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
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
#'   shap_backend = "treeshap",
#'   target_class = "1",
#'   n_models = 2,
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
#'   shap_backend = "kernelshap",
#'   pred_fun = pred_fun,
#'   target_class = "1",
#'   n_models = 2L,
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
#'   shap_backend = "shapr",
#'   pred_fun = pred_fun,
#'   target_class = "1",
#'   n_models = 2L,
#'   shapr_approach = "empirical",
#'   shapr_n_MC_samples = 1000L,
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
                    plot = TRUE,
                    performance_metric = "r2",
                    standardize_performance_metric = FALSE,
                    performance_type = "xval",
                    minimum_performance = 0,
                    method = "mean",
                    cutoff = 0.01,
                    top_n_features = NULL,
                    n_models = 2,
                    sample_size = NULL,
                    shap_backend = "auto",
                    model_performance = NULL,
                    target_class = NULL,
                    shapr_approach = "empirical",
                    shapr_phi0 = NULL,
                    shapr_n_MC_samples = 1000L,
                    n_core = -1L) {

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
  shap_backend <- match.arg(shap_backend,
                            c("auto", "h2o", "gpboost", "treeshap", "kernelshap", "shapr"))
  model_info <- shapley.models(models)

  # STEP 1: Keep the original H2O computation unchanged.
  # ============================================================
  if (model_info$h2o) {
    if (!requireNamespace("h2o", quietly = TRUE)) {
      stop("Package 'h2o' is required for H2O models. Install it with install.packages('h2o').")
    }
    if (!shap_backend %in% c("auto", "h2o")) {
      stop("H2O models require shap_backend = 'h2o' or 'auto'")
    }
    if (!is.null(model_performance)) {
      stop("H2O already retrieves performance estimates. Leave 'model_performance' as NULL.")
    }
    return(shapley.h2o(
      models = models, newdata = newdata, plot = plot,
      performance_metric = performance_metric,
      standardize_performance_metric = standardize_performance_metric,
      performance_type = performance_type,
      minimum_performance = minimum_performance,
      method = method, cutoff = cutoff,
      top_n_features = top_n_features,
      n_models = n_models, sample_size = sample_size,
      n_core = n_core
    ))
  }

  # STEP 2: Use native GPBoost SHAP, TreeSHAP, Kernel SHAP, or shapr.
  # ============================================================
  if (shap_backend == "h2o") stop("Only H2O models use the H2O backend")
  return(shapley.external(
    models = models, newdata = newdata, plot = plot,
    performance_metric = performance_metric,
    minimum_performance = minimum_performance,
    method = method, cutoff = cutoff,
    top_n_features = top_n_features,
    n_models = n_models, sample_size = sample_size,
    shap_backend = shap_backend,
    background_data = background_data,
    pred_fun = pred_fun,
    model_performance = model_performance,
    target_class = target_class,
    shapr_approach = shapr_approach,
    shapr_phi0 = shapr_phi0,
    shapr_n_MC_samples = shapr_n_MC_samples,
    n_core = n_core
  ))
}
