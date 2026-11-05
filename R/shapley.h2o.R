
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
                        n_core = 1L,
                        bootstrap = 0L) {

  if (!performance_metric %in% c("r2", "aucpr", "auc", "f2")) {
    stop("'performance_metric' must be 'r2', 'aucpr', or 'f2'")
  }

  if (!performance_type %in% c("train", "valid", "xval")) {
    stop("'performance_type' must be 'train', 'valid', or 'xval'")
  }

  if (!inherits(newdata, "H2OFrame")) stop("'newdata' must be an H2OFrame")

  # Identify H2O models
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
  if (length(ids) < n_models) stop("Fewer H2O models supplied than required by 'n_models'")

  # Sample observations once for all models
  # ============================================================
  if (!is.null(sample_size)) {

    if (length(sample_size) != 1L || !is.numeric(sample_size) ||
        !is.finite(sample_size) || sample_size < 1L ||
        sample_size > nrow(newdata) || sample_size != floor(sample_size)) {
      stop("Invalid 'sample_size'")
    }

    # Avoid unnecessary H2O frame creation when using all observations
    if (sample_size < nrow(newdata)) {
      newdata <- newdata[sample.int(nrow(newdata), sample_size), ]
    }
  }

  # Identify predictors across all models
  # ============================================================
  model_predictors <- lapply(ids, function(id) {
    as.character(h2o::h2o.getModel(id)@allparameters$x)
  })

  predictors <- unique(unlist(model_predictors, use.names = FALSE))

  if (!length(predictors) || !all(predictors %in% names(newdata))) {
    stop("Evaluation frame is missing predictors required by the H2O models")
  }

  # Compute SHAP values on the H2O server
  # ============================================================
  shared_data <- list(
    frame_id = h2o::h2o.getId(newdata),
    predictors = predictors,
    performance_metric = performance_metric,
    performance_type = performance_type,
    minimum_performance = minimum_performance
  )

  computed <- shapley.parallel(
    tasks = as.list(ids),
    worker = shapley.h2o.one,
    shared_data = shared_data,
    n_core = n_core,
    h2o = TRUE,
    dependencies = "shapley.h2o.one"
  )

  # Identify included models and calculate weights
  # ============================================================
  performances <- stats::setNames(vapply(computed, `[[`, numeric(1), "performance"), ids)
  included <- ids[performances > minimum_performance]

  ignored <- data.frame(
    id = ids[!ids %in% included],
    performance = as.numeric(performances[!ids %in% included]),
    stringsAsFactors = FALSE
  )

  if (length(included) < n_models) {
    stop("Number of models meeting 'minimum_performance' is lower than 'n_models'")
  }

  w <- shapley.weights(performances[included], performance_metric)
  selected_models <- computed[match(included, ids)]

  # Convert original SHAP contributions to R once
  # No additional H2O frames are required for bootstrapping.
  # ============================================================
  evaluation_data <- as.data.frame(newdata[, predictors])

  aggregated <- shapley.aggregate(
    results_by_model = selected_models,
    included = included,
    w = w,
    newdata = evaluation_data,
    method = method,
    cutoff = cutoff,
    top_n_features = top_n_features,
    bootstrap = bootstrap
  )

  # Return the WMSHAP object
  # ============================================================
  obj <- list(
    ids = ids,
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
    model_backends = stats::setNames(rep("h2o", length(included)), included),
    interval_type = "between-model descriptive interval",
    explanation_scope = "h2o_tree_prediction_raw",
    bootstrap_results = aggregated$bootstrap_results,
    bmv_results = aggregated$bmv_results,
    bootstrap_n = aggregated$bootstrap_n,
    bootstrap_interval_type = aggregated$bootstrap_interval_type,
    bmv_interval_type = aggregated$bmv_interval_type

  )

  class(obj) <- c("shapley", "list")

  if (plot) {
    obj$plot <- shapley.plot(obj, plot = "bar", method = method,
                             cutoff = cutoff, top_n_features = top_n_features)
  }

  return(obj)
}
