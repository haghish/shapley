
#' @title Bootstrap WMSHAP and Between-Model Variability
#'
#' @description
#' Estimates sampling variability in WMSHAP and between-model variability
#' (BMV) by resampling observations with replacement. Fitted models,
#' performance weights, and SHAP reference data remain fixed.
#'
#' @param results_by_model List of model-specific SHAP contributions.
#' @param w Numeric vector of model-performance weights.
#' @param bootstrap Integer. Number of bootstrap replications.
#' @param features Character vector of predictor names.
#'
#' @return A list containing bootstrap WMSHAP estimates, between-model
#'   standard deviations, pooled BMV, and percentile confidence intervals.
#' @author E. F. Haghish
#' @noRd

shapley.bootstrap <- function(results_by_model, w, bootstrap, features = NULL) {

  # Validate arguments
  # ============================================================
  if (!is.numeric(bootstrap) || length(bootstrap) != 1L || !is.finite(bootstrap) ||
      bootstrap < 2L || bootstrap != floor(bootstrap) || bootstrap > .Machine$integer.max) {
    stop("'bootstrap' must be an integer greater than or equal to 2", call. = FALSE)
  }

  m <- length(results_by_model)

  if (m < 2L || length(w) != m || any(!is.finite(w)) || any(w < 0) || sum(w) <= 0) {
    stop("Invalid model-performance weights for bootstrapping", call. = FALSE)
  }

  w <- w / sum(w)
  bootstrap <- as.integer(bootstrap)

  # Prepare absolute SHAP contributions once
  # ============================================================
  values <- lapply(results_by_model, function(z) abs(as.matrix(z$values)))
  n <- nrow(values[[1L]])
  p <- ncol(values[[1L]])

  if (n < 1L || p < 1L ||
      any(!vapply(values, function(z) identical(dim(z), c(n, p)), logical(1)))) {
    stop("All model-specific SHAP matrices must have matching dimensions", call. = FALSE)
  }

  if (is.null(features)) features <- colnames(values[[1L]])
  if (is.null(features)) features <- paste0("feature", seq_len(p))

  if (!is.character(features) || length(features) != p || anyNA(features) || anyDuplicated(features)) {
    stop("Invalid bootstrap feature names", call. = FALSE)
  }

  if (any(vapply(values, function(z) any(!is.finite(z)), logical(1)))) {
    stop("All SHAP contributions must be finite", call. = FALSE)
  }

  # Store weighted means and between-model SDs for every bootstrap sample
  # ============================================================
  replicates <- matrix(NA_real_, nrow = bootstrap, ncol = p, dimnames = list(NULL, features))
  bmv_replicates <- matrix(NA_real_, nrow = bootstrap, ncol = p, dimnames = list(NULL, features))

  # Bootstrap observations using identical indices for all models
  # ============================================================
  for (b in seq_len(bootstrap)) {

    counts <- tabulate(sample.int(n, size = n, replace = TRUE), nbins = n)

    weighted_mean <- numeric(p)
    weighted_square <- numeric(p)

    for (i in seq_len(m)) {

      importance <- as.numeric(crossprod(counts, values[[i]]))
      total <- sum(importance)

      if (!is.finite(total) || total <= 0) {
        stop("A bootstrap sample has zero total SHAP importance for model ",
             results_by_model[[i]]$id, call. = FALSE)
      }

      # Normalize importance ratios within each model and bootstrap sample.
      ratios <- importance / total

      # Calculate weighted first and second moments across models.
      weighted_mean <- weighted_mean + w[i] * ratios
      weighted_square <- weighted_square + w[i] * ratios^2
    }

    # Between-model variance and SD for this bootstrap sample.
    between_variance <- pmax(0, weighted_square - weighted_mean^2)

    replicates[b, ] <- weighted_mean
    bmv_replicates[b, ] <- sqrt(between_variance)
  }

  # Bootstrap percentile confidence intervals
  # ============================================================
  percentile <- function(x) {
    limits <- vapply(seq_len(ncol(x)), function(j) {
      as.numeric(stats::quantile(x[, j], probs = c(0.025, 0.975), names = FALSE))
    }, numeric(2))

    rownames(limits) <- c("lower", "upper")
    colnames(limits) <- colnames(x)
    return(limits)
  }

  mean_limits <- percentile(replicates)
  bmv_limits <- percentile(bmv_replicates)

  # Sampling variability of the WMSHAP mean
  # ============================================================
  bootstrap_mean <- colMeans(replicates)
  bootstrap_sd <- apply(replicates, 2L, stats::sd)

  # Root mean square of the bootstrap between-model SDs
  # ============================================================
  bmv_sd_mean <- colMeans(bmv_replicates)
  bmv_sd <- sqrt(colMeans(bmv_replicates^2))

  return(list(
    replicates = replicates,
    mean = stats::setNames(bootstrap_mean, features),
    sd = stats::setNames(bootstrap_sd, features),
    lowerCI = stats::setNames(mean_limits[1L, ], features),
    upperCI = stats::setNames(mean_limits[2L, ], features),

    bmv_replicates = bmv_replicates,
    bmv_sd = stats::setNames(bmv_sd, features),
    bmv_sd_mean = stats::setNames(bmv_sd_mean, features),
    bmv_sd_lowerCI = stats::setNames(bmv_limits[1L, ], features),
    bmv_sd_upperCI = stats::setNames(bmv_limits[2L, ], features)
  ))
}
