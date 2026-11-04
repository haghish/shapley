# Internal SHAPLEY function.
# Compute the native SHAP matrix and performance of one H2O model.
# ============================================================

shapley.h2o.one <- function(model_id, shared_data = NULL) {
  if (is.null(shared_data)) {
    shared_data <- get(".shapley_shared_data", envir = .GlobalEnv)
  }

  model <- h2o::h2o.getModel(model_id)
  metric <- shared_data$performance_metric
  train <- identical(shared_data$performance_type, "train")
  valid <- identical(shared_data$performance_type, "valid")
  xval <- identical(shared_data$performance_type, "xval")

  if (metric == "r2") {
    performance <- h2o::h2o.r2(model, train = train, valid = valid, xval = xval)
  } else if (metric == "aucpr") {
    performance <- h2o::h2o.aucpr(model, train = train, valid = valid, xval = xval)
  } else if (metric == "auc") {
    performance <- h2o::h2o.auc(model, train = train, valid = valid, xval = xval)
  } else if (metric == "f2") {
    performance <- h2o::h2o.F2(model)
  } else {
    stop("Unsupported performance metric")
  }
  performance <- as.numeric(performance)
  if (length(performance) != 1L || !is.finite(performance)) {
    stop("A single finite performance estimate is required for H2O model: ", model_id)
  }
  if (performance <= shared_data$minimum_performance) {
    return(list(id = model_id, performance = performance, values = NULL))
  }

  # The native contribution function returns one row per observation.
  # Unlike h2o.shap_summary_plot(), it does not construct a plot.
  # ============================================================
  test_hex <- h2o::h2o.getFrame(shared_data$frame_id)
  predictors <- shared_data$predictors
  shap <- h2o::h2o.predict_contributions(
    model, newdata = test_hex[, predictors], output_format = "compact"
  )
  shap <- as.data.frame(shap)

  # Exclude BiasTerm, and retain a common predictor matrix across models.
  # A predictor absent from the fitted model receives zero contribution.
  # ============================================================
  values <- matrix(0, nrow = nrow(shap), ncol = length(predictors),
                   dimnames = list(NULL, predictors))
  available <- intersect(predictors, names(shap))
  if (length(available)) {
    values[, available] <- as.matrix(shap[, available, drop = FALSE])
  }
  missing <- setdiff(as.character(model@allparameters$x), names(shap))
  if (length(missing)) {
    stop("H2O did not return compact SHAP contributions for predictors: ",
         paste(missing, collapse = ", "), ". Model: ", model_id)
  }
  if (any(!is.finite(values))) stop("Non-finite H2O SHAP values in model: ", model_id)

  return(list(id = model_id, performance = performance, values = values))
}
