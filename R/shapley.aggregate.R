# Internal SHAPLEY function.
# Global and local WMSHAP computed from the same SHAP matrices.
# Works with native H2O, TreeSHAP, Kernel SHAP, shapr, and GPBoost.
# ============================================================

shapley.aggregate <- function(results_by_model, included, w, newdata,
                              method = "mean", cutoff = 0.01,
                              top_n_features = NULL) {

  x <- names(newdata)
  n <- nrow(newdata)
  p <- length(x)
  m <- length(included)
  if (m < 2L || length(w) != m || length(results_by_model) != m) {
    stop("WMSHAP requires at least two models and matching performance weights")
  }
  if (!method %in% c("mean", "lowerCI")) {
    stop("'method' must be 'mean' or 'lowerCI'")
  }
  if (!is.null(top_n_features) &&
      (length(top_n_features) != 1L || !is.numeric(top_n_features) ||
       !is.finite(top_n_features) || top_n_features < 1 ||
       top_n_features != as.integer(top_n_features))) {
    stop("'top_n_features' must be a positive integer")
  }

  w <- w / sum(w)
  ratios <- matrix(NA_real_, nrow = p, ncol = m,
                   dimnames = list(x, included))
  importance <- matrix(NA_real_, nrow = p, ncol = m,
                       dimnames = list(x, included))
  local_mean <- matrix(0, nrow = n, ncol = p, dimnames = list(NULL, x))
  local_sq <- matrix(0, nrow = n, ncol = p, dimnames = list(NULL, x))
  shap_columns <- vector("list", m)

  # Use local SHAP values to compute global importance for each model.
  # ============================================================
  for (i in seq_len(m)) {
    if (!identical(results_by_model[[i]]$id, included[i])) {
      stop("SHAP results are not in the expected model order")
    }
    values <- as.matrix(results_by_model[[i]]$values)
    if (!identical(dim(values), c(n, p)) ||
        !identical(colnames(values), x) || any(!is.finite(values))) {
      stop("Invalid SHAP matrix for model: ", included[i])
    }
    importance[, i] <- colMeans(abs(values))
    total <- sum(importance[, i])
    if (total <= 0) stop("Zero total SHAP importance for model: ", included[i])
    ratios[, i] <- importance[, i] / total
    local_mean <- local_mean + w[i] * values
    local_sq <- local_sq + w[i] * values^2
    shap_columns[[i]] <- as.vector(t(values))
  }

  # Global WMSHAP ratios and between-model intervals.
  # ============================================================
  feature_importance <- setNames(lapply(seq_len(p), function(j) {
    importance[j, ]
  }), x)
  summaryShaps <- data.frame(feature = x, mean = NA_real_, sd = NA_real_,
                             ci = NA_real_, lowerCI = NA_real_,
                             upperCI = NA_real_, stringsAsFactors = FALSE)
  for (j in seq_len(p)) {
    summaryShaps[j, c("mean", "sd", "ci")] <- shapley.interval(ratios[j, ], w)
  }
  summaryShaps$lowerCI <- summaryShaps$mean - summaryShaps$ci
  summaryShaps$upperCI <- summaryShaps$mean + summaryShaps$ci

  # Individual WMSHAP contributions and between-model intervals.
  # ============================================================
  local_interval <- shapley.interval(w = w, mean = local_mean,
                                    mean_square = local_sq)
  localShaps <- data.frame(
    index = rep(seq_len(n), each = p),
    feature = rep(x, times = n),
    mean = as.vector(t(local_mean)),
    sd = as.vector(t(local_interval$sd)),
    ci = as.vector(t(local_interval$ci)),
    stringsAsFactors = FALSE
  )
  localShaps$lowerCI <- localShaps$mean - localShaps$ci
  localShaps$upperCI <- localShaps$mean + localShaps$ci

  # Retain the existing row-level data format for downstream methods.
  # ============================================================
  results <- data.frame(index = localShaps$index,
                        Row.names = seq_len(n * p),
                        feature = localShaps$feature,
                        stringsAsFactors = FALSE)
  names(shap_columns) <- c("contribution", paste0("contribution", seq_len(m)[-1L]))
  results <- cbind(results, as.data.frame(shap_columns, check.names = FALSE))

  selected <- summaryShaps[order(summaryShaps$mean, decreasing = TRUE), , drop = FALSE]
  if (!is.null(top_n_features)) {
    selected <- head(selected, top_n_features)
  } else if (method == "mean") {
    selected <- selected[selected$mean > cutoff, , drop = FALSE]
  } else {
    selected <- selected[!is.na(selected$lowerCI) &
                         selected$lowerCI > cutoff, , drop = FALSE]
  }

  # Use the same WMSHAP scatter plot for every model backend.
  # ============================================================
  values_numeric <- lapply(newdata[, x, drop = FALSE], function(z) {
    if (is.numeric(z)) as.numeric(z) else as.numeric(as.factor(z))
  })
  localShaps$value <- as.vector(t(as.matrix(as.data.frame(values_numeric))))
  contributionPlot <- ggplot2::ggplot(
    localShaps, ggplot2::aes(x = feature, y = mean, colour = value)
  ) + ggplot2::geom_point(
    position = ggplot2::position_jitter(width = 0.18), alpha = 0.6)

  return(list(summaryShaps = summaryShaps,
              localShaps = localShaps,
              selectedFeatures = selected$feature,
              feature_importance = feature_importance,
              results = results,
              contributionPlot = contributionPlot))
}
