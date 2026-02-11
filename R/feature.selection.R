#' @title Select top features by weighted mean SHAP (WMSHAP)
#' @description Selects a subset of features from a shapley object.
#'              Features can be selected by: (1) specified `features`,
#'              (2) `top_n_features`, or (3) WMSHAP cutoff for "mean" or "lowerCI".
#' @param shapley shapley object
#' @param method Character. Specifies statistic used for thresholding.
#'               Either \code{"mean"} or \code{"lowerCI"} (default) should be specified.
#'               Ignored if `top_n_features` is provided.
#' @param cutoff Numeric. Cutoff for thresholding on `method`.
#'               Default is zero, which means that all
#'               features with lower WMSHAP CI above zero will be selected.
#' @param top_n_features Integer. If provided, selects the top N features by `mean`,
#'                       overriding `method` and `cutoff`.
#' @param features Character vector of features to keep. If provided, it is applied
#'                 before `top_n_features`/`cutoff` selection (i.e., selection happens within this set).
#' @return A list with:
#' \describe{
#'   \item{shapley}{The updated shapley object.}
#'   \item{features}{Character vector of selected features, ordered by decreasing mean SHAP.}
#'   \item{mean}{Numeric vector of mean SHAP values aligned with `features`.}
#' }
#' @author E. F. Haghish



feature.selection <- function(shapley,
                              method = "lowerCI",
                              cutoff=0.0,
                              top_n_features=NULL,
                              features = NULL) {

  # Basic checks
  # ============================================================
  method <- match.arg(method)            # allow abbreviations
  if (is.null(shapley[["summaryShaps"]]) || is.null(shapley[["contributionPlot"]][["data"]])) {
    stop("shapley must include 'summaryShaps' and 'contributionPlot$data'", call. = FALSE)
  }

  # variables and feature selection
  # ============================================================
  DATA <- shapley$contributionPlot$data

  # if no cutoff or feature is specified, use all features. otherwise, select
  # the features based on the specified cutoff value or top_n_features
  if (is.null(features) & cutoff == 0) features <- as.character(shapley$summaryShaps$feature)

  # Select the features that meet the criteria
  # ============================================================
  if (length(shapley[["ids"]]) >= 1) {

    # Select the top N features
    if (!is.null(top_n_features)) {
      shapley$summaryShaps <- shapley$summaryShaps[order(
        shapley$summaryShaps$mean, decreasing = TRUE), ]
      shapley$summaryShaps <- shapley$summaryShaps[1:top_n_features, ]

      if (is.null(features) & cutoff > 0) {
        features <- as.character(shapley$summaryShaps$feature)
      }

      shapley$contributionPlot$data <- DATA[
        DATA$feature %in% features, ]

    }
    else if (method == "mean") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$mean > cutoff, ]
      if (is.null(features) & cutoff > 0) {
        features <- as.character(shapley$summaryShaps$feature)
      }
      shapley$contributionPlot$data <- DATA[DATA$feature %in% features, ]

    } else if (method == "lowerCI") {
      if (length(shapley[["ids"]]) == 1) stop("shapley object includes a single model and lowerCI cannot be used")
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$lowerCI > cutoff, ]
      if (is.null(features) & cutoff > 0) {
        features <- as.character(shapley$summaryShaps$feature)
      }
      shapley$contributionPlot$data <- DATA[DATA$feature %in% features, ]

    } else {
      stop("method must be one of 'mean' or 'lowerCI'")
    }
  }
  else (stop("at least 1 model must be included in the shapley object"))

  # Sort the features based on their mean SHAP values
  # ============================================================
  index <- order(- shapley$summaryShaps$mean)
  features <- features[index]
  mean <- shapley$summaryShaps$mean[index]

  return(list(shapley = shapley,
              features = features,
              mean = mean))

}


