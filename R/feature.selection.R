
#' @title Select top features by weighted mean SHAP (WMSHAP)
#' @description Selects a subset of features from a shapley object.
#'   Features can be selected by specifying a set of candidate features,
#'   selecting the top N features, or applying a WMSHAP cutoff to the
#'   mean or lower confidence interval.
#'
#' @param shapley A shapley object.
#' @param method Character. Statistic used for thresholding.
#'   Either \code{"mean"} (default) or \code{"lowerCI"}.
#'   Ignored if \code{top_n_features} is provided.
#' @param cutoff Numeric. Cutoff applied to the statistic specified
#'   by \code{method}. Only features exceeding this threshold are
#'   selected. The default is \code{0}.
#' @param top_n_features Integer or \code{NULL}. If provided, selects
#'   the top N features by mean WMSHAP, overriding \code{method}
#'   and \code{cutoff}.
#' @param features Character vector or \code{NULL}. Specifies the
#'   candidate features to consider. If provided, selection by
#'   \code{top_n_features} or \code{cutoff} is restricted to these
#'   features.
#'
#' @return A list containing:
#' \describe{
#'   \item{shapley}{The updated shapley object, containing only
#'     the selected features in \code{summaryShaps} and
#'     \code{contributionPlot$data}.}
#'   \item{features}{Character vector of selected features,
#'     ordered by decreasing mean WMSHAP.}
#'   \item{mean}{Numeric vector of mean WMSHAP values aligned
#'     with \code{features}.}
#' }
#' @author E. F. Haghish
#' @noRd


feature.selection <- function(shapley,
                              method = "mean",
                              cutoff = 0.0,
                              top_n_features = NULL,
                              features = NULL) {

  # Basic checks
  # ============================================================
  if (is.null(shapley[["summaryShaps"]]) ||
      is.null(shapley[["contributionPlot"]][["data"]])) {
    stop("shapley must include 'summaryShaps' and 'contributionPlot$data'", call. = FALSE)
  }

  if (length(shapley[["ids"]]) < 1L) {
    stop("at least 1 model must be included in the shapley object", call. = FALSE)
  }

  # Extract original data
  # ============================================================
  summary <- shapley$summaryShaps
  DATA <- shapley$contributionPlot$data

  # Restrict the analysis to specified candidate features
  # ============================================================
  if (!is.null(features)) {

    if (!is.character(features) || anyNA(features)) {
      stop("'features' must be a character vector", call. = FALSE)
    }

    summary <- summary[as.character(summary$feature) %in% features, , drop = FALSE]
  }

  # Select the top N features
  # ============================================================
  if (!is.null(top_n_features)) {

    if (!is.numeric(top_n_features) || length(top_n_features) != 1L ||
        !is.finite(top_n_features) || top_n_features < 1L ||
        top_n_features != floor(top_n_features)) {
      stop("'top_n_features' must be a positive integer", call. = FALSE)
    }

    summary <- summary[order(summary$mean, decreasing = TRUE), , drop = FALSE]
    summary <- utils::head(summary, top_n_features)

  } else {

    # Select features based on their mean or lower interval bound
    # ============================================================

    if (length(method) != 1L || is.na(method) ||
        !method %in% c("mean", "lowerCI", "bootstrap_lowerCI", "bmv_lower")) {
      stop("Invalid feature-selection method", call. = FALSE)
    }

    if (!is.numeric(cutoff) || length(cutoff) != 1L || !is.finite(cutoff)) {
      stop("'cutoff' must be a finite numeric value", call. = FALSE)
    }

    if (method == "lowerCI" && length(shapley[["ids"]]) == 1L) {
      stop("A single-model analysis cannot use lowerCI", call. = FALSE)
    }

    if (!method %in% names(summary)) {
      stop("The selected statistic is unavailable in summaryShaps", call. = FALSE)
    }

    statistic <- summary[[method]]
    keep <- !is.na(statistic) & statistic > cutoff
    summary <- summary[keep, , drop = FALSE]
  }

  # Sort selected features by decreasing mean WMSHAP
  # ============================================================
  summary <- summary[order(summary$mean, decreasing = TRUE), , drop = FALSE]

  selected_features <- as.character(summary$feature)
  selected_means <- summary$mean

  # Update the shapley object
  # ============================================================
  shapley$summaryShaps <- summary
  shapley$contributionPlot$data <- DATA[DATA$feature %in% selected_features, , drop = FALSE]

  return(list(
    shapley = shapley,
    features = selected_features,
    mean = selected_means
  ))
}
