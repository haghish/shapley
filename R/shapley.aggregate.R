
#' @title Aggregate SHAP Values Across Multiple Models
#'
#' @description Computes global and local WMSHAP values, descriptive
#'   between-model intervals, observation-bootstrap confidence intervals,
#'   and Between-Model Variability (BMV) statistics.
#'
#' @author E. F. Haghish
#' @noRd

shapley.aggregate <- function(results_by_model,
                              included,
                              w,
                              newdata,
                              method = "mean",
                              cutoff = 0.01,
                              top_n_features = NULL,
                              bootstrap = 0L) {

  x <- names(newdata)
  n <- nrow(newdata)
  p <- length(x)
  m <- length(included)

  # Validate arguments
  # ============================================================
  if (m < 2L || length(w) != m || length(results_by_model) != m) {
    stop("WMSHAP requires at least two models and matching performance weights")
  }

  if (n < 1L || p < 1L) stop("'newdata' must contain observations and predictors")
  if (any(!is.finite(w)) || any(w < 0) || sum(w) <= 0) stop("Invalid model-performance weights")

  if (!method %in% c("mean", "lowerCI", "bootstrap_lowerCI", "bmv_lower")) {
    stop("'method' must be 'mean', 'lowerCI', 'bootstrap_lowerCI', or 'bmv_lower'")
  }

  if (!is.null(top_n_features) &&
      (length(top_n_features) != 1L || !is.numeric(top_n_features) ||
       !is.finite(top_n_features) || top_n_features < 1L ||
       top_n_features != floor(top_n_features))) {
    stop("'top_n_features' must be a positive integer")
  }

  if (!is.numeric(bootstrap) || length(bootstrap) != 1L || !is.finite(bootstrap) ||
      bootstrap < 0L || bootstrap == 1L || bootstrap != floor(bootstrap)) {
    stop("'bootstrap' must be 0 or an integer greater than or equal to 2")
  }

  if (method %in% c("bootstrap_lowerCI", "bmv_lower") && bootstrap == 0L) {
    stop("The selected method requires bootstrap >= 2")
  }

  w <- w / sum(w)

  # Initialize matrices
  # ============================================================
  ratios <- matrix(NA_real_, nrow = p, ncol = m, dimnames = list(x, included))
  importance <- matrix(NA_real_, nrow = p, ncol = m, dimnames = list(x, included))
  local_mean <- matrix(0, nrow = n, ncol = p, dimnames = list(NULL, x))
  local_sq <- matrix(0, nrow = n, ncol = p, dimnames = list(NULL, x))
  shap_columns <- vector("list", m)

  # Compute global importance and local WMSHAP
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

  # Global WMSHAP ratios and descriptive between-model intervals
  # ============================================================
  feature_importance <- stats::setNames(lapply(seq_len(p), function(j) importance[j, ]), x)

  summaryShaps <- data.frame(feature = x, mean = NA_real_, sd = NA_real_,
                             ci = NA_real_, lowerCI = NA_real_,
                             upperCI = NA_real_, stringsAsFactors = FALSE)

  for (j in seq_len(p)) {
    summaryShaps[j, c("mean", "sd", "ci")] <- shapley.interval(ratios[j, ], w)
  }

  summaryShaps$lowerCI <- summaryShaps$mean - summaryShaps$ci
  summaryShaps$upperCI <- summaryShaps$mean + summaryShaps$ci

  # Bootstrap sampling variability and BMV
  # ============================================================
  boot <- NULL

  if (bootstrap > 0L) {

    boot <- shapley.bootstrap(results_by_model, w, bootstrap, features = x)

    # Sampling variability in the weighted mean.
    summaryShaps$bootstrap_mean <- unname(boot$mean[x])
    summaryShaps$bootstrap_sd <- unname(boot$sd[x])
    summaryShaps$bootstrap_lowerCI <- unname(boot$lowerCI[x])
    summaryShaps$bootstrap_upperCI <- unname(boot$upperCI[x])

    # BMV: pooled between-model variability across bootstrap samples.
    summaryShaps$bmv_sd <- unname(boot$bmv_sd[x])
    summaryShaps$bmv_sd_mean <- unname(boot$bmv_sd_mean[x])
    summaryShaps$bmv_sd_lowerCI <- unname(boot$bmv_sd_lowerCI[x])
    summaryShaps$bmv_sd_upperCI <- unname(boot$bmv_sd_upperCI[x])

    # Descriptive BMV interval centered on the original WMSHAP estimate.
    summaryShaps$bmv_ci <- 1.96 * summaryShaps$bmv_sd
    summaryShaps$bmv_lower <- pmax(0, summaryShaps$mean - summaryShaps$bmv_ci)
    summaryShaps$bmv_upper <- pmin(1, summaryShaps$mean + summaryShaps$bmv_ci)
  }

  # Local WMSHAP contributions and descriptive between-model intervals
  # ============================================================
  local_interval <- shapley.interval(w = w, mean = local_mean, mean_square = local_sq)

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

  # Retain existing row-level SHAP results
  # ============================================================
  results <- data.frame(index = localShaps$index, Row.names = seq_len(n * p),
                        feature = localShaps$feature, stringsAsFactors = FALSE)

  names(shap_columns) <- c("contribution", paste0("contribution", seq_len(m)[-1L]))
  results <- cbind(results, as.data.frame(shap_columns, check.names = FALSE))

  # Feature selection
  # ============================================================
  selected <- summaryShaps[order(summaryShaps$mean, decreasing = TRUE), , drop = FALSE]

  if (!is.null(top_n_features)) {
    selected <- utils::head(selected, top_n_features)
  } else {
    statistic <- selected[[method]]
    selected <- selected[!is.na(statistic) & statistic > cutoff, , drop = FALSE]
  }

  # Plot observation-level WMSHAP contributions in H2O summary plot style
  # ============================================================
  # Normalize feature values to [0, 1] for the colour gradient.
  # Numeric predictors use empirical ranks; categorical predictors use
  # their factor levels. Missing values are shown at the low end of the scale.
  normalize_value <- function(z) {
    if (is.factor(z) || is.character(z) || is.logical(z)) {
      z <- as.factor(z)
      value <- (as.numeric(z) - 1) / max(1, nlevels(z) - 1)
      value[is.na(value)] <- 0
      return(value)
    }

    z <- as.numeric(z)
    valid <- is.finite(z)
    value <- numeric(length(z))
    if (any(valid)) value[valid] <- stats::ecdf(z[valid])(z[valid])
    return(value)
  }

  values_numeric <- lapply(newdata[, x, drop = FALSE], normalize_value)
  localShaps$value <- as.vector(t(as.matrix(as.data.frame(values_numeric))))

  # Place the most important features at the top of the rotated plot.
  plotData <- localShaps
  feature_order <- summaryShaps$feature[order(summaryShaps$mean)]
  plotData$feature <- factor(plotData$feature, levels = feature_order)

  contributionPlot <- ggplot2::ggplot(
    plotData,
    ggplot2::aes(x = !!as.name("feature"), y = !!as.name("mean"),
                 colour = !!as.name("value"))
  ) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::geom_point(position = shapley.jitter(width = 0.4), alpha = 0.5) +
    ggplot2::scale_colour_gradient(low = "#00AAEE", high = "#FF1166") +
    ggplot2::coord_flip() +
    ggplot2::ggtitle("") +
    ggplot2::xlab("Features\n") +
    ggplot2::ylab("\nWeighted Mean SHAP contributions") +
    ggplot2::labs(colour = "Normalized values") +
    ggplot2::theme_classic() +
    ggplot2::theme(
      legend.position = "top",
      legend.justification = "right",
      legend.title.align = 0.5,
      legend.direction = "horizontal",
      legend.text = ggplot2::element_text(colour = "black", size = 6, face = "bold"),
      legend.key.height = grid::unit(0.3, "cm"),
      legend.key.width = grid::unit(1, "cm"),
      legend.margin = ggplot2::margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"),
      plot.margin = ggplot2::margin(t = -0.5, r = 0.25, b = 0.25, l = 0.25,
                                    unit = "cm")
    ) +
    ggplot2::guides(colour = ggplot2::guide_colourbar(
      title.position = "top", title.hjust = 0.5
    ))

  return(list(
    summaryShaps = summaryShaps,
    localShaps = localShaps,
    selectedFeatures = selected$feature,
    feature_importance = feature_importance,
    results = results,
    contributionPlot = contributionPlot,

    bootstrap_results = if (!is.null(boot)) boot$replicates else NULL,
    bmv_results = if (!is.null(boot)) boot$bmv_replicates else NULL,
    bootstrap_n = as.integer(bootstrap),
    bootstrap_interval_type = if (!is.null(boot)) "observation bootstrap percentile CI" else NULL,
    bmv_interval_type = if (!is.null(boot)) "descriptive bootstrap-pooled between-model variability" else NULL
  ))
}
