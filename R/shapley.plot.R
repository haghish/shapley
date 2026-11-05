
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
#' @param interval Interval type to display: \code{"auto"}, \code{"bmv"},
#'   \code{"bootstrap"}, and \code{"between_model"}.
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
                         labels = NULL,
                         interval = "auto") {

  # Validate arguments
  # ============================================================
  if (!inherits(shapley, "shapley")) stop("shapley object must be of class 'shapley'")

  if (plot == "shap") {
    plot <- "wmshap"
    warning("plot = 'shap' is deprecated; use plot = 'wmshap'.", call. = FALSE)
  }

  if (!plot %in% c("bar", "wmshap")) stop("plot must be 'bar' or 'wmshap'")

  if (!method %in% c("mean", "lowerCI", "bootstrap_lowerCI", "bmv_lower")) {
    stop("Invalid feature-selection method", call. = FALSE)
  }

  interval <- match.arg(interval, c("auto", "bmv", "bootstrap", "between_model", "both", "all"))

  has_bootstrap <- all(c("bootstrap_lowerCI", "bootstrap_upperCI") %in% names(shapley$summaryShaps))
  has_bmv <- all(c("bmv_lower", "bmv_upper") %in% names(shapley$summaryShaps))

  if (interval == "auto") interval <- if (has_bmv) "bmv" else "between_model"

  if (interval %in% c("bmv", "all") && !has_bmv) {
    stop("BMV is unavailable. Enable bootstrapping in shapley().", call. = FALSE)
  }

  if (interval %in% c("bootstrap", "both", "all") && !has_bootstrap) {
    stop("Bootstrap intervals are unavailable. Enable bootstrapping in shapley().", call. = FALSE)
  }

  # Select features
  # ============================================================
  select <- feature.selection(shapley = shapley, method = method, cutoff = cutoff,
                              top_n_features = top_n_features, features = features)

  shapley <- select$shapley

  if (nrow(shapley$summaryShaps) == 0L) {
    stop("No features meet the selection criteria", call. = FALSE)
  }

  # Plot global WMSHAP ratios
  # ============================================================
  if (plot == "bar") {

    summary <- shapley$summaryShaps
    summary$feature <- factor(summary$feature, levels = summary$feature[order(summary$mean)])

    Plot <- ggplot2::ggplot(summary, ggplot2::aes(x = !!as.name("feature"), y = !!as.name("mean"))) +
      ggplot2::geom_col(fill = "#07B86B", alpha = 0.8) +
      ggplot2::coord_flip() +
      ggplot2::xlab("Features\n") +
      ggplot2::ylab("\nWeighted Mean SHAP Ratio") +
      ggplot2::theme_classic()

    # Determine which intervals should be displayed
    # ============================================================
    show_bmv <- has_bmv && interval %in% c("bmv", "both", "all")
    show_bootstrap <- interval %in% c("bootstrap", "both", "all")
    show_between <- interval %in% c("between_model", "all") ||
      (interval == "both" && !has_bmv)

    # Original between-model interval
    # ============================================================
    if (show_between) {
      Plot <- Plot + ggplot2::geom_errorbar(
        ggplot2::aes(ymin = !!as.name("lowerCI"), ymax = !!as.name("upperCI")),
        width = 0.38, color = "#7A004BF0", alpha = 0.75, linewidth = 0.7
      )
    }

    # Bootstrap-pooled Between-Model Variability (BMV)
    # ============================================================
    if (show_bmv) {
      Plot <- Plot + ggplot2::geom_errorbar(
        ggplot2::aes(ymin = !!as.name("bmv_lower"), ymax = !!as.name("bmv_upper")),
        width = 0.25, color = "#147D64", alpha = 0.9, linewidth = 0.9
      )
    }

    # Bootstrap confidence interval for the WMSHAP mean
    # ============================================================
    if (show_bootstrap) {
      Plot <- Plot + ggplot2::geom_errorbar(
        ggplot2::aes(ymin = !!as.name("bootstrap_lowerCI"), ymax = !!as.name("bootstrap_upperCI")),
        width = 0.12, color = "#225DA8", linewidth = 0.9
      )
    }

    # Describe the displayed intervals without labeling BMV as a CI
    # ============================================================
    descriptions <- character(0)

    if (show_bmv) descriptions <- c(descriptions, "Green: BMV interval")
    if (show_bootstrap) descriptions <- c(descriptions, "Blue: bootstrap 95% CI")
    if (show_between) descriptions <- c(descriptions, "Purple: original between-model interval")

    Plot <- Plot + ggplot2::labs(subtitle = paste(descriptions, collapse = "; "))

  } else {

    # Plot observation-level weighted mean SHAP contributions
    # ============================================================
    selected_features <- as.character(shapley$summaryShaps$feature)
    Plot <- shapley$contributionPlot

    if (!is.null(shapley$localShaps)) {
      Plot$data <- shapley$localShaps[
        shapley$localShaps$feature %in% selected_features, , drop = FALSE
      ]
    }

    Plot$data$feature <- factor(as.character(Plot$data$feature), levels = rev(selected_features))

    Plot <- Plot +
      ggplot2::coord_flip() +
      ggplot2::ggtitle("") +
      ggplot2::xlab("Features\n") +
      ggplot2::ylab("\nWeighted Mean SHAP contributions") +
      ggplot2::theme_classic() +
      ggplot2::labs(colour = if (is.null(shapley$localShaps)) "Normalized values" else "Feature value") +
      ggplot2::theme(legend.position = "top")

    if (length(scale_colour_gradient) == 3L) {
      Plot <- Plot + ggplot2::scale_colour_gradient2(
        low = scale_colour_gradient[1], mid = scale_colour_gradient[2],
        high = scale_colour_gradient[3], midpoint = 0.5
      )
    }
  }

  if (!is.null(labels)) Plot <- Plot + ggplot2::scale_x_discrete(labels = labels)

  print(Plot)
  return(Plot)
}
