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
#' @param feature character vector, specifying the feature to be plotted.
#' @author E. F. Haghish
#' @return normalized numeric vector
#' @export

shapley.feature.selection <- function(shapley,
                                      method = "lowerCI",
                                      cutoff=0.0,
                                      top_n_features=NULL,
                                      features = NULL) {

  # Select the features that meet the criteria
  # ============================================================
  if (!is.null(top_n_features)) {
    shapley$summaryShaps <- shapley$summaryShaps[order(
      shapley$summaryShaps$mean, decreasing = TRUE), ]
    shapley$summaryShaps <- shapley$summaryShaps[1:top_n_features, ]

    shapley$contributionPlot$data <- shapley$contributionPlot$data[
      shapley$contributionPlot$data$feature %in% shapley$summaryShaps$feature, ]
  }
  else if (method == "mean") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$mean > cutoff, ]
      shapley$contributionPlot$data <- shapley$contributionPlot[
        shapley$contributionPlot$feature %in% shapley$summaryShaps$feature, ]
  } else if (method == "shapratio") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$shapratio > cutoff, ]
      shapley$contributionPlot$data <- shapley$contributionPlot$data[
        shapley$contributionPlot$data$feature %in% shapley$summaryShaps$feature, ]
  } else if (method == "lowerCI") {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$lowerCI > cutoff, ]
      shapley$contributionPlot$data <- shapley$contributionPlot$data[
        shapley$contributionPlot$data$feature %in% shapley$summaryShaps$feature, ]
  } else if (!is.null(features)) {
      shapley$summaryShaps <- shapley$summaryShaps[shapley$summaryShaps$feature %in% features, ]
      shapley$contributionPlot$data <- shapley$contributionPlot$data[
        shapley$contributionPlot$data$feature %in% shapley$summaryShaps$feature, ]
  } else {
    stop("method must be one of 'mean', 'shapratio', or 'ci'")
  }

  # Sort the features based on their mean SHAP values
  # ============================================================
  index <- order(- shapley$summaryShaps$mean)
  features <- shapley$summaryShaps$feature[index]
  mean <- shapley$summaryShaps$mean[index]
  shapratio <- shapley$summaryShaps$shapratio[index]

  return(list(shapley = shapley,
              features = features,
              mean = mean,
              shapratio = shapratio))

}
