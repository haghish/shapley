# Internal SHAPLEY function.
# Normalize model-performance scores for all SHAP backends.
# ============================================================

shapley.weights <- function(performance, metric) {
  score <- as.numeric(performance)
  if (length(score) < 2L || any(!is.finite(score))) {
    stop("At least two finite model-performance estimates are required")
  }
  if (metric %in% c("aucpr", "auc", "f2")) {
    if (any(score < 0 | score > 1)) {
      stop("Classification performance estimates must be between 0 and 1")
    }
  }

  # Negative R-squared cannot be used directly as a positive weight.
  # Shift the scores by their minimum, retaining their order.
  # ============================================================
  if (metric == "r2" && any(score < 0)) {
    score <- score - min(score) + sqrt(.Machine$double.eps)
  }

  if (any(score < 0) || sum(score) <= 0) {
    stop("Model performance does not define positive weights")
  }
  if (sum(score > 0) < 2L) {
    stop("At least two models must have positive performance weights")
  }

  weights <- score / sum(score)
  names(weights) <- names(performance)
  return(weights)
}
