#' @title Normalize Model-Performance Weights
#'
#' @description Normalizes model-performance scores to obtain weights
#'   for computing Weighted Mean SHAP (WMSHAP) values. The weights are
#'   scaled to sum to one. Negative r2 are not allowed.
#'
#' @param performance Numeric vector containing the performance scores
#'   of the selected models.
#' @param metric Character. Performance metric used to evaluate the
#'   models. Supported metrics are \code{"r2"}, \code{"auc"},
#'   \code{"aucpr"}, and \code{"f2"}.
#'
#' @return A named numeric vector of normalized model-performance
#'   weights that sum to one.
#' @noRd

shapley.weights <- function(performance, metric) {
  score <- as.numeric(performance)

  # Syntax processing
  if (length(score) < 2L || any(!is.finite(score))) {
    stop("At least two finite model-performance estimates are required")
  }
  if (metric %in% c("aucpr", "auc", "f2")) {
    if (any(score < 0 | score > 1)) {
      stop("Classification performance estimates must be between 0 and 1")
    }
  }
  if (metric == "r2" && any(score < 0)) stop("Model performances include negative r2")
  if (any(score < 0) || sum(score) <= 0) stop("Model performance does not define positive weights")
  if (sum(score > 0) < 2L) stop("At least two models must have positive performance weights")

  # normalizing the weights
  weights <- score / sum(score)
  names(weights) <- names(performance)
  return(weights)
}
