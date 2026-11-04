#' @title Plot Global WMSHAP Values
#' @description Plot global feature importance using Weighted Mean SHAP ratios
#'   or plot the individual-level weighted mean SHAP contributions.
#' @param shapley An object returned by \code{shapley()}.
#' @param plot Plot type: \code{"bar"} or \code{"wmshap"}.
#' @param method Feature selection method: \code{"mean"} or \code{"lowerCI"}.
#' @param cutoff Minimum global WMSHAP ratio or lower confidence interval.
#' @param top_n_features Optional number of features to plot.
#' @param features Optional character vector of features to plot.
#' @param legendstyle Legend type used for the contribution plot.
#' @param scale_colour_gradient Optional colour vector for the contribution plot.
#' @param labels Optional feature labels.
#' @return A ggplot object.
#' @export
shapley.plot <- function(shapley,
                         plot = "bar",
                         method = "mean",
                         cutoff = 0.01,
                         top_n_features = NULL,
                         features = NULL,
                         legendstyle = "continuous",
                         scale_colour_gradient = NULL,
                         labels = NULL) {

  if (!inherits(shapley, "shapley")) stop("shapley object must be of class 'shapley'")
  if (plot == "shap") {
    plot <- "wmshap"
    warning("plot = 'shap' is deprecated; use plot = 'wmshap'.", call. = FALSE)
  }
  if (!plot %in% c("bar", "wmshap")) stop("plot must be 'bar' or 'wmshap'")
  if (!method %in% c("mean", "lowerCI")) stop("method must be 'mean' or 'lowerCI'")

  # STEP 1: Select the features to be plotted.
  # ============================================================
  if (is.null(shapley$localShaps)) {
    select <- feature.selection(shapley = shapley, method = method,
                                cutoff = cutoff, top_n_features = top_n_features,
                                features = features)
    shapley <- select$shapley
  }
  else {
    summaryShaps <- shapley$summaryShaps
    if (!is.null(features)) {
      summaryShaps <- summaryShaps[summaryShaps$feature %in% features, , drop = FALSE]
    }
    else if (!is.null(top_n_features)) {
      summaryShaps <- head(summaryShaps[order(summaryShaps$mean, decreasing = TRUE), , drop = FALSE],
                           top_n_features)
    }
    else if (method == "lowerCI") {
      summaryShaps <- summaryShaps[!is.na(summaryShaps$lowerCI) &
                                   summaryShaps$lowerCI > cutoff, , drop = FALSE]
    }
    else {
      summaryShaps <- summaryShaps[summaryShaps$mean > cutoff, , drop = FALSE]
    }
    shapley$summaryShaps <- summaryShaps
  }
  if (nrow(shapley$summaryShaps) == 0L) stop("No features meet the selection criteria")

  # STEP 2: Plot the global WMSHAP ratios.
  # ============================================================
  if (plot == "bar") {
    shapley$summaryShaps$feature <- factor(
      shapley$summaryShaps$feature,
      levels = shapley$summaryShaps$feature[order(shapley$summaryShaps$mean)])
    ftr <- shapley$summaryShaps$feature
    nrmm <- shapley$summaryShaps$mean
    lci <- shapley$summaryShaps$lowerCI
    uci <- shapley$summaryShaps$upperCI
    Plot <- ggplot2::ggplot(data = NULL,
                            ggplot2::aes(x = ftr, y = nrmm)) +
      ggplot2::geom_col(fill = "#07B86B", alpha = 0.8) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lci, ymax = uci),
                             width = 0.2, color = "#7A004BF0",
                             alpha = 0.75, linewidth = 0.7) +
      ggplot2::coord_flip() +
      ggplot2::xlab("Features\n") +
      ggplot2::ylab("\nWeighted Mean SHAP Ratio with 95% CI") +
      ggplot2::theme_classic()
  }

  # STEP 3: Plot the local WMSHAP contributions.
  # ============================================================
  else {
    if (is.null(shapley$localShaps)) {
      # Original H2O plot; the data are contained in contributionPlot.
      shapley$contributionPlot$data$feature <- factor(
        shapley$summaryShaps$feature,
        levels = shapley$summaryShaps$feature[order(shapley$summaryShaps$mean)])
      Plot <- shapley$contributionPlot
    }
    else {
      # The other backends store the performance-weighted contributions in localShaps.
      selected <- as.character(shapley$summaryShaps$feature)
      data <- shapley$localShaps[shapley$localShaps$feature %in% selected, , drop = FALSE]
      data$feature <- factor(data$feature,
                             levels = selected[order(shapley$summaryShaps$mean)])
      # Reuse the contribution plot stored in the shapley object.
      Plot <- shapley$contributionPlot
      Plot$data <- data
      Plot <- Plot + ggplot2::coord_flip()
    }
    Plot <- Plot + ggplot2::ggtitle("") + ggplot2::xlab("Features\n") +
      ggplot2::ylab("\nWeighted Mean SHAP contributions") +
      ggplot2::theme_classic() +
      ggplot2::labs(colour = if (is.null(shapley$localShaps)) "Normalized values" else "Feature value") +
      ggplot2::theme(legend.position = "top")
    if (length(scale_colour_gradient) == 3L) {
      Plot <- Plot + ggplot2::scale_colour_gradient2(
        low = scale_colour_gradient[1], mid = scale_colour_gradient[2],
        high = scale_colour_gradient[3], midpoint = 0.5)
    }
  }
  if (!is.null(labels)) Plot <- Plot + ggplot2::scale_x_discrete(labels = labels)
  print(Plot)
  return(Plot)
}
