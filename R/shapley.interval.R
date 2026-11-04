# Internal SHAPLEY function.
# ============================================================

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

  # Summarize between-model variation.
  # ============================================================
  if (length(w) < 2L || effective_n <= 1 + 1e-10) {
    sd <- mean * NA_real_
    ci <- mean * NA_real_
  }
  else {
    variance <- pmax(0, mean_square - mean^2) / (1 - sum(w^2))
    sd <- sqrt(variance)
    ci <- 1.96 * sd / sqrt(effective_n)
  }

  if (is.null(dim(mean))) return(c(mean = mean, sd = sd, ci = ci))
  return(list(mean = mean, sd = sd, ci = ci))
}
