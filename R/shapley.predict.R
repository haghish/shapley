# Internal SHAPLEY function.
# ============================================================

shapley.predict <- function(model, newdata, pred_fun = NULL,
                                    target_class = NULL, model_id = NULL) {
  n <- nrow(newdata)

  # Use the prediction function defined by the user, when provided.
  # ============================================================
  if (!is.null(pred_fun)) {
    prediction <- pred_fun(model, newdata)
  }
  else if (inherits(model, "ranger")) {
    prediction <- predict(model, data = newdata)$predictions
  }
  else if (inherits(model, "randomForest")) {
    if (identical(model$type, "classification")) {
      prediction <- stats::predict(model, newdata = newdata, type = "prob")
    } else {
      prediction <- stats::predict(model, newdata = newdata)
    }
  }
  else if (inherits(model, "glm")) {
    prediction <- stats::predict(model, newdata = newdata, type = "response")
  }
  else if (inherits(model, "gpb.Booster")) {
    if (any(!vapply(newdata, is.numeric, logical(1)))) {
      stop("GPBoost requires numeric predictors or a custom 'pred_fun'.")
    }
    prediction <- stats::predict(model, data = as.matrix(newdata),
                                 pred_latent = TRUE, ignore_gp_model = TRUE)
  }
  else if (inherits(model, "xgb.Booster") || inherits(model, "lgb.Booster")) {
    if (!all(vapply(newdata, is.numeric, logical(1)))) {
      stop("For this boosting model, supply numeric predictors or specify 'pred_fun'.")
    }
    prediction <- predict(model, as.matrix(newdata))
  }
  else {
    prediction <- stats::predict(model, newdata = newdata)
  }

  # Choose one target if several predictions are returned for each row.
  # ============================================================
  if (is.data.frame(prediction) || is.matrix(prediction)) {
    prediction <- as.matrix(prediction)
    if (nrow(prediction) != n) stop("Unexpected number of predictions for model ", model_id)
    if (ncol(prediction) == 1L) {
      prediction <- prediction[, 1]
    } else {
      if (is.null(target_class)) {
        if (ncol(prediction) != 2L) {
          stop("Specify 'target_class' for a model with multiple outputs")
        }
        k <- 2L
      } else if (!is.null(colnames(prediction)) &&
                 as.character(target_class) %in% colnames(prediction)) {
        k <- match(as.character(target_class), colnames(prediction))
      } else if (is.numeric(target_class) && length(target_class) == 1L &&
                 target_class >= 1 && target_class <= ncol(prediction)) {
        k <- as.integer(target_class)
      } else {
        stop("'target_class' does not match the prediction columns for model ", model_id)
      }
      prediction <- prediction[, k]
    }
  }
  if (is.factor(prediction)) {
    stop("Predictions are factors. Specify 'pred_fun' returning numeric predictions.")
  }
  if (!is.numeric(prediction) || length(prediction) != n ||
      any(!is.finite(prediction))) {
    stop("The prediction function must return one finite numeric value per row for model ", model_id)
  }
  return(as.numeric(prediction))
}
