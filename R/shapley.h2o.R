#' @title Compute Weighted Mean SHAP Values for H2O Models
#'
#' @description
#' Computes Weighted Mean SHAP (WMSHAP) values and associated intervals
#' across multiple H2O models, including models trained through H2O grid
#' search, AutoML, or stacked ensembles.
#'
#' @details
#' SHAP contributions are obtained directly from H2O using
#' \code{h2o::h2o.predict_contributions()}. For each model, the absolute
#' SHAP contributions are averaged across observations and normalized
#' to obtain feature importance ratios. These ratios are then combined
#' across models using normalized performance weights to calculate
#' WMSHAP values and associated between-model intervals.
#'
#' The function also calculates performance-weighted SHAP contributions
#' for individual observations. All models are evaluated on the same
#' observations to ensure that their contributions are comparable.
#' Models with performance estimates at or below the specified minimum
#' are excluded from the analysis.
#'
#' Important features can be selected based on their mean WMSHAP ratios,
#' the lower bounds of their intervals, or a specified number of
#' top-ranked features. SHAP contributions can be computed in parallel
#' across models to improve computational efficiency.
#'
#' @param models An H2O AutoML object, H2O grid object, \code{autoEnsemble}
#'   object, or character vector of H2O model IDs. At least
#'   \code{n_models} models must be available for the analysis.
#'
#' @param newdata An \code{H2OFrame} containing the observations used to
#'   compute SHAP contributions. The data must contain all predictors
#'   required by the supplied models.
#'
#' @param plot Logical. If \code{TRUE} (default), generates a bar plot
#'   of global WMSHAP ratios and their intervals using
#'   \code{shapley.plot()}.
#'
#' @param performance_metric Character. Performance metric used to
#'   weight models. Available options are \code{"r2"} (default),
#'   \code{"aucpr"}, \code{"auc"}, and \code{"f2"}. The latter three
#'   are intended for classification models.
#'
#' @param performance_type Character. Specifies which model performance
#'   estimates are used for weighting. Available options are
#'   \code{"train"}, \code{"valid"}, and \code{"xval"} (default),
#'   corresponding to training, validation, and cross-validation
#'   performance, respectively. The requested estimates must be
#'   available for the fitted models.
#'
#' @param minimum_performance Numeric. Minimum model performance
#'   required for inclusion in the analysis. Models with performance
#'   estimates at or below this threshold are excluded. The default
#'   is \code{0}.
#'
#' @param method Character. Method used for selecting important features.
#'   Available options are \code{"mean"} (default), which selects
#'   features based on their mean WMSHAP ratios, and \code{"lowerCI"},
#'   which uses the lower bounds of their between-model intervals.
#'
#' @param cutoff Numeric. WMSHAP cutoff used to identify and select
#'   important features. The recommended threshold is 1 divided by
#'   the number of predictors in the model. The default is \code{0.01}.
#'
#' @param top_n_features Integer or \code{NULL}. Number of top-ranked
#'   features to select based on their mean WMSHAP ratios. If
#'   \code{NULL} (default), features are selected according to
#'   \code{method} and \code{cutoff}. When specified, this argument
#'   overrides the cutoff-based selection.
#'
#' @param n_models Integer. Minimum number of models required for
#'   computing WMSHAP values after excluding models that do not meet
#'   the performance threshold. The default is \code{10}. This
#'   argument does not limit the number of models included.
#'
#' @param sample_size Integer or \code{NULL}. Number of observations
#'   randomly sampled from \code{newdata} for computing SHAP values.
#'   If \code{NULL} (default), all observations are used. The same
#'   sampled observations are evaluated across all models.
#'
#' @param n_core Integer. Number of parallel R workers used to compute
#'   SHAP contributions across models. The default is \code{1},
#'   which performs the computations sequentially. A positive integer
#'   specifies the number of workers, whereas \code{-1} uses all
#'   detected physical CPU cores, limited by the number of models.
#'
#' @return An object of class \code{"shapley"}
#'
#'
#' @keywords internal
#' @author E. F. Haghish
#' @noRd

shapley.h2o <- function(models,
                        newdata,
                        plot = TRUE,
                        performance_metric = "r2",
                        performance_type = "xval",
                        minimum_performance = 0,
                        method = "mean",
                        cutoff = 0.01,
                        top_n_features = NULL,
                        n_models = 10,
                        sample_size = NULL,
                        n_core = 1L) {

  if (!performance_metric %in% c("r2", "aucpr", "auc", "f2", "mcc")) {
    stop("'performance_metric' must be 'r2', 'aucpr', 'auc', 'mcc', or 'f2'")
  }
  if (!performance_type %in% c("train", "valid", "xval")) {
    stop("'performance_type' must be 'train', 'valid', or 'xval'")
  }
  if (!inherits(newdata, "H2OFrame")) stop("'newdata' must be an H2OFrame")

  # Identify all H2O models.
  # ============================================================
  if (inherits(models, c("H2OAutoML", "H2OGrid"))) {
    ids <- h2o.get_ids(models)
  } else if (inherits(models, "autoEnsemble")) {
    ids <- models[["top_rank_id"]]
  } else if (is.character(models)) {
    ids <- models
  } else {
    stop("'models' must contain H2O models or model IDs")
  }
  ids <- as.character(ids)
  if (anyDuplicated(ids)) stop("H2O model IDs must be unique")
  if (length(ids) < n_models) {
    stop("Fewer H2O models supplied than required by 'n_models'")
  }

  # Sample rows once for all models to ensure aligned local contributions.
  # ============================================================
  if (!is.null(sample_size)) {
    if (length(sample_size) != 1L || !is.numeric(sample_size) ||
        !is.finite(sample_size) || sample_size < 1L ||
        sample_size > nrow(newdata) || sample_size != as.integer(sample_size)) {
      stop("Invalid 'sample_size'")
    }
    newdata <- newdata[sample.int(nrow(newdata), sample_size), ]
  }

  # All H2O models must use the same predictor names for valid WMSHAP.
  # A union of predictors also allows some models to omit features.
  # ============================================================
  model_predictors <- lapply(ids, function(id) {
    as.character(h2o::h2o.getModel(id)@allparameters$x)
  })
  predictors <- unique(unlist(model_predictors, use.names = FALSE))
  if (!length(predictors) || !all(predictors %in% names(newdata))) {
    stop("Evaluation frame is missing predictors required by the H2O models")
  }

  # Each worker retrieves one model's contributions from the same H2O server.
  # ============================================================
  shared_data <- list(frame_id = h2o::h2o.getId(newdata),
                      predictors = predictors,
                      performance_metric = performance_metric,
                      performance_type = performance_type,
                      minimum_performance = minimum_performance)
  computed <- shapley.parallel(
    tasks = as.list(ids), worker = shapley.h2o.one,
    shared_data = shared_data, n_core = n_core,
    h2o = TRUE, dependencies = "shapley.h2o.one"
  )

  performances <- setNames(vapply(computed, `[[`, numeric(1), "performance"), ids)
  included <- ids[performances > minimum_performance]
  ignored <- data.frame(id = ids[!ids %in% included],
                        performance = as.numeric(performances[!ids %in% included]),
                        stringsAsFactors = FALSE)
  if (length(included) < n_models) {
    stop("Number of models meeting 'minimum_performance' is lower than 'n_models'")
  }

  # The same normalization is used by all model backends.
  # ============================================================
  w <- shapley.weights(performances[included], performance_metric)
  selected_models <- computed[match(included, ids)]
  evaluation_data <- as.data.frame(newdata[, predictors])
  aggregated <- shapley.aggregate(selected_models, included, w,
                                  newdata = evaluation_data,
                                  method = method, cutoff = cutoff,
                                  top_n_features = top_n_features)

  # Preserve legacy output weights for older domain and feature methods.
  # The calculation itself uses normalized weights that sum to one.
  # ============================================================
  obj <- list(ids = ids,
              plot = NULL,
              contributionPlot = aggregated$contributionPlot,
              summaryShaps = aggregated$summaryShaps,
              selectedFeatures = aggregated$selectedFeatures,
              feature_importance = aggregated$feature_importance,
              weights = unname(w * length(w)),
              model_weights = w,
              model_performance = performances,
              results = aggregated$results,
              shap_contributions_by_ids = aggregated$results,
              ignored_models = ignored,
              included_models = included,
              localShaps = aggregated$localShaps,
              backend = "h2o",
              model_backends = setNames(rep("h2o", length(included)), included),
              interval_type = "between-model descriptive interval",
              explanation_scope = "h2o_tree_prediction_raw")
  class(obj) <- c("shapley", "list")
  if (plot) {
    obj$plot <- shapley.plot(obj, plot = "bar", method = method,
                             cutoff = cutoff, top_n_features = top_n_features)
  }
  return(obj)
}
