#' @title Selects the top features with highest weighted mean shap values based on the
#'        specified criteria
#' @description This function specifies the top features and prepares the data
#'              for plotting SHAP contributions for each row, or summary of absolute
#'              SHAP contributions for each feature.
#' @param shapley shapley object
#' @param method character, specifying the method used for identifying the most
#'               important features according to their weighted SHAP values.
#'               The default selection method is "lowerCI", which includes
#'               features whose lower weighted confidence interval exceeds the
#'               predefined 'cutoff' value (default is relative SHAP of 1%).
#'               Alternatively, the "mean" option can be specified, indicating
#'               any feature with normalized weighted mean SHAP contribution above
#'               the specified 'cutoff' should be selected. Another
#'               alternative options is "shapratio", a method that filters
#'               for features where the proportion of their relative weighted SHAP
#'               value exceeds the 'cutoff'. This approach calculates the relative
#'               contribution of each feature's weighted SHAP value against the
#'               aggregate of all features, with those surpassing the 'cutoff'
#'               being selected as top feature.
#' @param cutoff numeric, specifying the cutoff for the method used for selecting
#'               the top features. the default is zero, which means that all
#'               features with the "method" criteria above zero will be selected.
#' @param top_n_features integer. if specified, the top n features with the
#'                       highest weighted SHAP values will be selected, overrullung
#'                       the 'cutoff' and 'method' arguments.
#' @param features character vector, specifying the feature to be plotted.
#' @author E. F. Haghish
#' @return normalized numeric vector
#' @export

shapley.feature.selection <- function(shapley,
                                      method = "lowerCI",
                                      cutoff=0.0,
                                      top_n_features=NULL,
                                      features = NULL) {

  # variables
  # ============================================================
  DATA <- shapley$contributionPlot$data
  if (is.null(features)) features <- as.character(shapley$summaryShaps$feature)

  # Select the features that meet the criteria
  # ============================================================
  if (length(shapley[["ids"]]) >= 1) {
    if (!is.null(top_n_features)) {
      shapley$summaryShaps <- shapley$summaryShaps[order(
        shapley$summaryShaps$mean, decreasing = TRUE), ]
      shapley$summaryShaps <- shapley$summaryShaps[1:top_n_features, ]

      shapley$contributionPlot$data <- DATA[
        DATA$feature %in% features, ]
    }
    else if (method == "mean") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$mean > cutoff, ]
      shapley$contributionPlot$data <- DATA[DATA$feature %in% features, ]

    } else if (method == "shapratio") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$shapratio > cutoff, ]
      shapley$contributionPlot$data <- DATA[DATA$feature %in% features, ]

    } else if (method == "lowerCI") {
      if (length(shapley[["ids"]]) == 1) stop("shapley object includes a single model and lowerCI cannot be used")
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$lowerCI > cutoff, ]
      shapley$contributionPlot$data <- DATA[DATA$feature %in% features, ]

    } else {
      stop("method must be one of 'mean', 'shapratio', or 'ci'")
    }
  }
  else (stop("at least 1 model must be included in the shapley object"))

  # Sort the features based on their mean SHAP values
  # ============================================================
  index <- order(- shapley$summaryShaps$mean)
  features <- features[index]
  mean <- shapley$summaryShaps$mean[index]
  shapratio <- shapley$summaryShaps$shapratio[index]

  return(list(shapley = shapley,
              features = features,
              mean = mean,
              shapratio = shapratio))

}


