#' @title Compute SHAP Values for GPBoost Models
#'
#' @description Computes SHAP values for the boosted-tree component of
#'   a fitted GPBoost model using its native SHAP implementation.
#'   Random effects and Gaussian process contributions are not included.
#'
#' @param model A fitted GPBoost model of class \code{"gpb.Booster"}.
#' @param newdata Data frame containing the observations and numeric
#'   predictors for which SHAP values are computed.
#' @param model_id Optional model identifier used in error messages.
#'
#' @return A numeric matrix containing SHAP values for each observation
#'   and predictor.
#'
#' @noRd


shapley.gpboost <- function(model, newdata, model_id = NULL) {

  if (!inherits(model, "gpb.Booster")) {
    stop("The 'gpboost' backend requires a fitted 'gpb.Booster' object")
  }
  if (!requireNamespace("gpboost", quietly = TRUE)) {
    stop("The 'gpboost' backend requires the optional package 'gpboost'. ",
         "Install it with install.packages('gpboost').")
  }
  if (utils::packageVersion("gpboost") < "1.7.4") {
    stop("The 'gpboost' backend requires gpboost >= 1.7.4.")
  }
  if (any(!vapply(newdata, is.numeric, logical(1)))) {
    stop("GPBoost SHAP requires numeric predictors in 'newdata'")
  }

  # GPBoost computes TreeSHAP from the tree ensemble only.
  # ignore_gp_model ensures random effects are not included in the prediction.
  # ============================================================
  x <- as.matrix(newdata)
  contributions <- stats::predict(
    model, data = x, predcontrib = TRUE,
    ignore_gp_model = TRUE, reshape = TRUE
  )

  # The last column contains the baseline (expected prediction).
  # Multiple-output models cannot be explained safely without selecting an output.
  # ============================================================
  if (!is.matrix(contributions) || !is.numeric(contributions) ||
      nrow(contributions) != nrow(newdata) ||
      ncol(contributions) != ncol(x) + 1L) {
    stop("Unexpected GPBoost SHAP output for model ", model_id,
         ". Expected one matrix column per predictor plus one baseline column. ",
         "Multi-output predictions are not supported by the native gpboost backend.")
  }

  feature_shap <- contributions[, seq_len(ncol(x)), drop = FALSE]
  colnames(feature_shap) <- colnames(x)
  if (any(!is.finite(feature_shap))) {
    stop("Non-finite GPBoost SHAP values in model ", model_id)
  }
  return(feature_shap)
}
