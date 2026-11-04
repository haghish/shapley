# Internal SHAPLEY function.
# ============================================================

shapley.external <- function(models, newdata, plot, performance_metric,
                             minimum_performance, method, cutoff, top_n_features,
                             n_models, sample_size, shap_backend, background_data,
                             pred_fun, model_performance, target_class,
                             shapr_approach = "empirical", shapr_phi0 = NULL,
                             shapr_n_MC_samples = 1000L,
                             n_core = 1L) {

  if (!is.data.frame(newdata) && !is.matrix(newdata)) {
    stop("For non-H2O models, 'newdata' must be a data frame or matrix")
  }
  newdata <- as.data.frame(newdata)
  if (anyDuplicated(names(newdata)) || is.null(names(newdata))) {
    stop("'newdata' must have unique predictor names")
  }
  if (!is.null(sample_size)) {
    if (length(sample_size) != 1L || !is.finite(sample_size) ||
        sample_size < 1 || sample_size > nrow(newdata) || sample_size != as.integer(sample_size)) {
      stop("Invalid 'sample_size'")
    }
    newdata <- newdata[sample.int(nrow(newdata), sample_size), , drop = FALSE]
  }
  if (nrow(newdata) < 1L || ncol(newdata) < 1L) stop("'newdata' must contain predictors and observations")
  if (is.null(background_data)) {
    # Native GPBoost SHAP uses the fitted trees rather than background data.
    # Do not warn or change the random seed for a GPBoost-only analysis.
    # ============================================================
    gpboost_only <- all(vapply(shapley.models(models)$models,
                              function(z) inherits(z, "gpb.Booster"), logical(1)))
    if (gpboost_only) {
      background_data <- newdata[seq_len(min(200L, nrow(newdata))), , drop = FALSE]
    } else {
      # Use one common background, not a new random sample for every model.
      # A representative training dataset is preferable to this fallback.
      warning("No 'background_data' supplied. Using up to 200 rows of 'newdata' as a common reference dataset.")
      rows <- sample.int(nrow(newdata), min(200L, nrow(newdata)))
      background_data <- newdata[rows, , drop = FALSE]
    }
  } else {
    background_data <- as.data.frame(background_data)
    if (!all(names(newdata) %in% names(background_data))) {
      stop("'background_data' must contain all the predictors in 'newdata'")
    }
    background_data <- background_data[, names(newdata), drop = FALSE]
    if (nrow(background_data) < 1L) stop("'background_data' must have observations")
  }
  if (!is.null(pred_fun) && !is.function(pred_fun)) stop("'pred_fun' must be a function")
  if (!performance_metric %in% c("r2", "aucpr", "auc", "f2")) {
    stop("Invalid 'performance_metric'")
  }
  if (!method %in% c("mean", "lowerCI")) stop("Invalid 'method'")
  if (identical(shap_backend, "shapr")) {
    if (!is.character(shapr_approach) || !length(shapr_approach)) {
      stop("'shapr_approach' must be a character vector")
    }
    if (length(shapr_n_MC_samples) != 1L ||
        !is.finite(shapr_n_MC_samples) || shapr_n_MC_samples < 1L) {
      stop("'shapr_n_MC_samples' must be a positive integer")
    }
  }
  if (!is.numeric(minimum_performance) || length(minimum_performance) != 1L) {
    stop("'minimum_performance' must be numeric")
  }

  model_list <- shapley.models(models)$models
  ids <- names(model_list)
  performances <- shapley.performance(model_performance, ids)
  included <- ids[performances > minimum_performance]
  ignored <- data.frame(id = ids[!ids %in% included],
                        performance = unname(performances[!ids %in% included]),
                        stringsAsFactors = FALSE)
  if (length(included) < n_models) {
    stop("Number of models meeting the minimum performance is lower than 'n_models'")
  }
  if (length(included) == 0L) stop("No models were retained")

  w <- shapley.weights(performances[included], performance_metric)
  m <- length(included)
  # Determine the SHAP backend for each fitted model.
  # ============================================================
  backend <- vapply(included, function(id) {
    shapley.detect(model_list[[id]], shap_backend,
                   background_data, target_class = target_class)
  }, character(1))
  names(backend) <- included
  if (length(unique(backend)) > 1L) {
    if (identical(shap_backend, "auto")) {
      if ("gpboost" %in% backend) {
        stop("Automatic WMSHAP cannot combine GPBoost tree-only SHAP ",
             "with other backends. Select a common method and scale explicitly.")
      }
      message("Different model classes detected; using kernelshap for all models.")
      backend[] <- "kernelshap"
    } else {
      stop("Models use different SHAP backends; select one common backend")
    }
  }

  # Validate optional backend dependencies on demand.
  # ============================================================
  if (any(backend == "shapr") && !requireNamespace("shapr", quietly = TRUE)) {
    stop("The optional package 'shapr' is required. Install it with install.packages('shapr').")
  }
  if (any(backend == "shapr") && utils::packageVersion("shapr") < "1.1.0") {
    stop("The 'shapr' backend requires shapr >= 1.1.0")
  }
  gpboost_models <- any(vapply(model_list[included],
                              function(z) inherits(z, "gpb.Booster"), logical(1)))
  if ((any(backend == "gpboost") || gpboost_models) &&
      !requireNamespace("gpboost", quietly = TRUE)) {
    stop("The optional package 'gpboost' is required. Install it with install.packages('gpboost').")
  }
  if ((any(backend == "gpboost") || gpboost_models) &&
      utils::packageVersion("gpboost") < "1.7.4") {
    stop("GPBoost explanations require gpboost >= 1.7.4")
  }

  # Prepare one task per retained model.
  # Every model is explained once, independently of its WMSHAP weight.
  # ============================================================
  tasks <- lapply(seq_along(included), function(i) {
    id <- included[i]
    phi0 <- NULL
    if (!is.null(shapr_phi0)) {
      if (length(shapr_phi0) == 1L) {
        phi0 <- shapr_phi0
      } else if (!is.null(names(shapr_phi0))) {
        if (!id %in% names(shapr_phi0)) stop("Missing shapr_phi0 for model ", id)
        phi0 <- shapr_phi0[[id]]
      } else if (length(shapr_phi0) == m) {
        phi0 <- shapr_phi0[i]
      } else {
        stop("'shapr_phi0' must be a scalar or one value per included model")
      }
    }
    list(id = id, model = model_list[[id]], backend = unname(backend[i]),
         phi0 = phi0)
  })

  # GPBoost models use R6 objects with native pointers. For parallel
  # computation, save each model once and reload it inside its R worker.
  # This also supports boosters with a fitted GPModel component.
  # ============================================================
  if (shapley.cores(n_core) > 1L && length(tasks) > 1L &&
      any(vapply(tasks, function(z) inherits(z$model, "gpb.Booster"), logical(1)))) {
    gp_dir <- tempfile("shapley_gpboost_")
    dir.create(gp_dir)
    on.exit(unlink(gp_dir, recursive = TRUE, force = TRUE), add = TRUE)
    for (i in seq_along(tasks)) {
      if (inherits(tasks[[i]]$model, "gpb.Booster")) {
        model_file <- file.path(gp_dir, paste0("model_", i, ".json"))
        gpboost::gpb.save(tasks[[i]]$model, filename = model_file)
        tasks[[i]]$model <- NULL
        tasks[[i]]$model_file <- model_file
      }
    }
  }

  # Run SHAP computations across different models in parallel.
  # The selected backend computes each model's SHAP values independently.
  # ============================================================
  shared_data <- list(newdata = newdata, background_data = background_data,
                      pred_fun = pred_fun, target_class = target_class,
                      shapr_approach = shapr_approach,
                      shapr_n_MC_samples = shapr_n_MC_samples)

  results_by_model <- shapley.parallel(
    tasks = tasks,
    worker = shapley.external.one,
    shared_data = shared_data,
    n_core = n_core,
    dependencies = c("shapley.external.one", "shapley.compute",
                     "shapley.choose.class", "shapley.predict", "shapley.gpboost")
  )

  # One common WMSHAP procedure for all learners, including H2O.
  # ============================================================
  aggregated <- shapley.aggregate(results_by_model, included, w,
                                  newdata = newdata, method = method,
                                  cutoff = cutoff,
                                  top_n_features = top_n_features)
  summaryShaps <- aggregated$summaryShaps
  localShaps <- aggregated$localShaps
  selected <- aggregated$selectedFeatures
  feature_importance <- aggregated$feature_importance
  results <- aggregated$results
  contributionPlot <- aggregated$contributionPlot
  weights <- unname(w * length(w))  # legacy output scale (sum = model count)

  obj <- list(ids = ids, plot = NULL, contributionPlot = contributionPlot,
              summaryShaps = summaryShaps,
              selectedFeatures = selected,
              feature_importance = feature_importance,
              weights = weights, model_weights = setNames(w, included),
              model_performance = performances,
              results = results,
              shap_contributions_by_ids = results,
              ignored_models = ignored, included_models = included,
              localShaps = localShaps, backend = unique(backend),
              model_backends = backend, target_class = target_class,
              interval_type = "between-model descriptive interval",
              explanation_scope = if (any(vapply(model_list[included],
                function(z) inherits(z, "gpb.Booster"), logical(1)))) {
                if (is.null(pred_fun)) "gpboost_tree_component" else "user_defined_prediction"
              } else "model_prediction",
              shapr_approach = if (all(backend == "shapr")) shapr_approach else NULL)
  class(obj) <- c("shapley", "list")

  if (plot) {
    obj$plot <- shapley.plot(obj, plot = "bar", method = method,
                                    cutoff = cutoff, top_n_features = top_n_features)
  }
  return(obj)
}
