#' @title Compute Weighted Intervals Across Model Set
#'
#' @description Computes the weighted mean, descriptive population
#'   standard deviation, and descriptive interval half-width of SHAP values
#'   across multiple models using normalized model-performance weights.
#'
#' @param values Optional numeric vector containing the SHAP values
#'   or feature importance estimates from individual models.
#' @param w Numeric vector of model-performance weights. Weights are
#'   normalized to sum to one.
#' @param mean Optional numeric value, vector, or matrix containing
#'   precomputed weighted means.
#' @param mean_square Optional numeric value, vector, or matrix containing
#'   precomputed weighted means of squared values.
#'
#' @return A named numeric vector containing \code{mean}, \code{sd},
#'   and \code{ci}, or a list containing these components when
#'   matrix inputs are provided. The \code{ci} component is a descriptive
#'   half-width (1.96 times the SD divided by the square root of the
#'   effective number of models), not a confidence interval.
#'
#' @noRd


shapley.interval <- function(values = NULL, w, mean = NULL, mean_square = NULL) {

  # The weights are normalized independently of the performance metric.
  # ============================================================
  w <- w / sum(w)
  effective_n <- 1 / sum(w^2)

  if (!is.null(values)) {
    if (length(values) != length(w)) stop("Values and weights have different lengths")
    mean <- sum(w * values)
    mean_square <- sum(w * values^2)
  }
  if (is.null(mean) || is.null(mean_square)) {
    stop("Specify values, or both mean and mean_square")
  }

  # Summarize descriptive between-model variation.
  # ============================================================
  if (length(w) < 2L || effective_n <= 1 + 1e-10) {
    sd <- mean * NA_real_
    ci <- mean * NA_real_
  }
  else {
    variance <- pmax(0, mean_square - mean^2)
    sd <- sqrt(variance)
    ci <- 1.96 * sd / sqrt(effective_n)
  }

  if (is.null(dim(mean))) return(c(mean = mean, sd = sd, ci = ci))
  return(list(mean = mean, sd = sd, ci = ci))
}
