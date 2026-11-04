#' @title Plot Individual-Level WMSHAP Values
#' @description Computes and plots model-performance-weighted SHAP contributions
#'   and between-model intervals for an individual observation.
#' @param shapley An object returned by \code{shapley()}.
#' @param row_index Index of the observation to explain.
#' @param top_n_features Optional number of important predictors to plot.
#' @param features Optional predictor names.
#' @param nonzeroCI If \code{TRUE}, show only contributions whose intervals
#'   do not contain zero.
#' @param plot Logical. If \code{TRUE}, print the plot.
#' @param print Logical. If \code{TRUE}, print the results table.
#' @return A list containing the summary table and a ggplot object.
#' @export
shapley.row.plot <- function(shapley,
                             row_index,
                             top_n_features = NULL,
                             features = NULL,
                             nonzeroCI = FALSE,
                             plot = TRUE,
                             print = FALSE) {

  if (!inherits(shapley, "shapley")) stop("shapley object must be of class 'shapley'")
  if (length(row_index) != 1L || !is.numeric(row_index)) {
    stop("'row_index' must be a single numeric index")
  }

  # STEP 1: Obtain the individual-level WMSHAP summary.
  # ============================================================
  if (!is.null(shapley$localShaps)) {
    rowSummary <- shapley$localShaps
    rowSummary <- rowSummary[rowSummary$index == row_index,
                             c("feature", "mean", "sd", "ci", "lowerCI", "upperCI"),
                             drop = FALSE]
    if (nrow(rowSummary) == 0L) stop("Unknown 'row_index'")
    if (!is.null(features)) {
      rowSummary <- rowSummary[rowSummary$feature %in% features, , drop = FALSE]
    }
  }
  else {
    # Original H2O procedure.
    w <- shapley$weights
    results <- shapley$results
    UNQ <- unique(results$feature)
    if (!is.null(features)) UNQ <- UNQ[UNQ %in% features]
    rowSummary <- data.frame(feature = UNQ, mean = NA, sd = NA, ci = NA,
                             lowerCI = NA, upperCI = NA)
    results <- results[results$index == row_index, ]
    for (j in UNQ) {
      tmp <- results[results$feature == j,
                     grep("^contribution", names(results))]
      weighted_mean <- stats::weighted.mean(tmp, w, na.rm = TRUE)
      weighted_var <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE) /
        (sum(w, na.rm = TRUE) - 1)
      weighted_sd <- sqrt(weighted_var)
      ci <- 1.96 * weighted_sd / sqrt(length(tmp))
      rowSummary[rowSummary$feature == j, "mean"] <- weighted_mean
      rowSummary[rowSummary$feature == j, "sd"] <- weighted_sd
      rowSummary[rowSummary$feature == j, "ci"] <- ci
      rowSummary[rowSummary$feature == j, "lowerCI"] <- weighted_mean - ci
      rowSummary[rowSummary$feature == j, "upperCI"] <- weighted_mean + ci
    }
  }

  # STEP 2: Apply the feature selection criteria.
  # ============================================================
  if (!is.null(top_n_features) && is.null(features)) {
    rowSummary <- rowSummary[order(abs(rowSummary$mean), decreasing = TRUE), ]
    rowSummary <- head(rowSummary, top_n_features)
  }
  if (nonzeroCI) {
    onedirection <- !is.na(rowSummary$lowerCI) &
      ((rowSummary$lowerCI > 0 & rowSummary$upperCI > 0) |
         (rowSummary$lowerCI < 0 & rowSummary$upperCI < 0))
    rowSummary <- rowSummary[onedirection, , drop = FALSE]
  }
  if (nrow(rowSummary) == 0L) stop("No features meet the selection criteria")

  # STEP 3: Create the plot.
  # ============================================================
  ftr <- rowSummary$feature
  MEAN <- rowSummary$mean
  lci <- rowSummary$lowerCI
  uci <- rowSummary$upperCI
  Plot <- ggplot2::ggplot(data = NULL,
                          ggplot2::aes(x = ftr, y = MEAN)) +
    ggplot2::geom_col(fill = "#07B86B", alpha = 0.8) +
    ggplot2::geom_hline(yintercept = 0, linetype = "solid",
                        color = "gray70", alpha = 0.75, linewidth = 0.7) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = lci, ymax = uci),
                           width = 0.2, color = "#7A004BF0",
                           alpha = 0.75, linewidth = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::ylab(paste("\nWeighted Mean SHAP contributions for row", row_index)) +
    ggplot2::xlab("Features\n") + ggplot2::theme_classic()
  Plot$rowSummarySHAP <- rowSummary
  if (plot) base::print(Plot)
  if (print) base::print(rowSummary)
  return(list(summary = rowSummary, plot = Plot))
}
