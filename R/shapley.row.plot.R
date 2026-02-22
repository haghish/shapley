#' @title WMSHAP row-level plot for a single observation (participant or data row)
#' @description
#' Computes and visualizes Weighted Mean SHAP contributions (WMSHAP) for a single row
#' (subject/observation) across multiple models in a \code{shapley} object.
#' For each feature, the function computes a weighted mean of row-level SHAP contributions
#' across models using \code{shapley$weights} and reports an approximate 95% confidence
#' interval summarizing variability across models.
#' @param shapley object of class \code{"shapley"}, as returned by the 'shapley' function
#' @param row_index Integer (length 1). The row/subject identifier to visualize. This is
#'                  matched against the \code{index} column in \code{shapley$results}.
#' @param top_n_features Integer. If specified, the top n features with the
#'                       highest weighted SHAP values will be selected. This
#'                       will be overrulled by the 'features' argument.
#' @param features Optional character vector of feature names to plot. If \code{NULL},
#'                 all available features in \code{shapley$results} are used.
#'                 Specifying the \code{features} argument will override the
#'                 \code{top_n_features} argument.
#' @param nonzeroCI Logical. If \code{TRUE}, it avoids ploting features that have
#'                  a confidence interval crossing zero.
#' @param plot Logical. If \code{TRUE}, prints the plot.
#' @param print Logical. If \code{TRUE}, prints the computed summary table for the row.
#' @return a list including the GGPLOT2 object and the data frame of WMSHAP summary values.
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
#' @importFrom h2o h2o.stackedEnsemble h2o.getModel h2o.auc h2o.aucpr h2o.r2
#'             h2o.F2 h2o.mean_per_class_error h2o.giniCoef h2o.accuracy
#'             h2o.shap_summary_plot
# @importFrom h2otools h2o.get_ids
#' @importFrom curl curl
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
#'             geom_hline
#' @examples
#'
#' \dontrun{
#' # load the required libraries for building the base-learners and the ensemble models
#' library(h2o)            #shapley supports h2o models
#' library(shapley)
#'
#' # initiate the h2o server
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE,
#'          insecure = TRUE)
#'
#' # upload data to h2o cloud
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#'
#' set.seed(10)
#'
#' ### H2O provides 2 types of grid search for tuning the models, which are
#' ### AutoML and Grid. Below, I demonstrate how weighted mean shapley values
#' ### can be computed for both types.
#'
#' #######################################################
#' ### EXAMPLE 1: PREPARE AutoML Grid (takes a couple of minutes)
#' #######################################################
#' # run AutoML to tune various models (GBM) for 60 seconds
#' y <- "CAPSULE"
#' prostate[,y] <- as.factor(prostate[,y])  #convert to factor for classification
#' aml <- h2o.automl(y = y, training_frame = prostate, max_runtime_secs = 120,
#'                  include_algos=c("GBM"),
#'
#'                  seed = 2023, nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' ### call 'shapley' function to compute the weighted mean and weighted confidence intervals
#' ### of SHAP values across all trained models.
#' ### Note that the 'newdata' should be the testing dataset!
#' result <- shapley(models = aml, newdata = prostate,
#'                   performance_metric = "aucpr", plot = TRUE)
#'
#' shapley.row.plot(result, row_index = 11)
#'
#' #######################################################
#' ### EXAMPLE 2: PREPARE H2O Grid (takes a couple of minutes)
#' #######################################################
#' # make sure equal number of "nfolds" is specified for different grids
#' grid <- h2o.grid(algorithm = "gbm", y = y, training_frame = prostate,
#'                  hyper_params = list(ntrees = seq(1,50,1)),
#'                  grid_id = "ensemble_grid",
#'
#'                  # this setting ensures the models are comparable for building a meta learner
#'                  seed = 2023, fold_assignment = "Modulo", nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' result2 <- shapley(models = grid, newdata = prostate,
#'                    performance_metric = "aucpr", plot = TRUE)
#'
#' shapley.row.plot(result2, row_index = 9)
#' shapley.row.plot(result2, row_index = 9, nonzeroCI = TRUE)
#' shapley.row.plot(result2, row_index = 9, top_n_features = 10)
#'
#' #######################################################
#' ### EXAMPLE 3: PREPARE autoEnsemble STACKED ENSEMBLE MODEL
#' #######################################################
#'
#' ### get the models' IDs from the AutoML and grid searches.
#' ### this is all that is needed before building the ensemble,
#' ### i.e., to specify the model IDs that should be evaluated.
#' library(autoEnsemble)
#' ids    <- c(h2o.get_ids(aml), h2o.get_ids(grid))
#' autoSearch <- ensemble(models = ids, training_frame = prostate, strategy = "search")
#' result3 <- shapley(models = autoSearch, newdata = prostate,
#'                    performance_metric = "aucpr", plot = TRUE)
#'
#' #plot all important features
#' shapley.row.plot(result3, row_index = 13)
#'
#' #plot only the given features
#' shapPlot <- shapley.row.plot(result3, row_index = 13, features = c("PSA", "AGE"))
#'
#' # inspect the computed data for the row 13
#' ptint(shapPlot$summary)
#' }
#' @author E. F. Haghish
#' @export

shapley.row.plot <- function(shapley,
                             row_index,
                             top_n_features = NULL,
                             features = NULL,
                             nonzeroCI = FALSE,
                             plot = TRUE,
                             print = FALSE) {

  # Variable definitions
  # ============================================================
  w <- shapley$weights

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")

  if (length(row_index) > 1) stop("'row_index' should have a length of 1")

  # Get the data of the participant (row)
  #     If only one row is selected, return the raw SHAP
  #     If multiple rows are selected, return the absolute SHAP for that subset
  # ============================================================
  results <- shapley$results
  #rowname <- paste0(row_index, ".")
  UNQ     <- unique(results$feature)
  if (!is.null(features)) UNQ <- UNQ[UNQ %in% features]
  rowSummary <- data.frame(
    feature = UNQ,
    mean = NA,
    sd = NA,
    ci = NA,
    lowerCI = NA,
    upperCI = NA)

  results <- results[results$index == row_index, ]

  # compute WMSHAP for the row
  # ============================================================
  for (j in UNQ) {
    tmp <- results[results$feature == j,
                           grep("^contribution", names(results))]

    weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
    weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
    weighted_sd   <- sqrt(weighted_var)
    ci            <- 1.96 * weighted_sd / sqrt(length(tmp))

    rowSummary[rowSummary$feature == j, "mean"] <- weighted_mean
    rowSummary[rowSummary$feature == j, "sd"]   <- weighted_sd
    rowSummary[rowSummary$feature == j, "ci"]   <- ci

    # Compute the lower and upper confidence intervals
    # -------------------------------------------------------------
    rowSummary[rowSummary$feature == j, "lowerCI"] <- weighted_mean - ci
    rowSummary[rowSummary$feature == j, "upperCI"] <- weighted_mean + ci
  }

  # subset the row
  # ============================================================
  if (!is.null(top_n_features) & is.null(features)) {
    rowSummary <- rowSummary[order(abs(rowSummary$mean), decreasing = TRUE), ]
    rowSummary <- rowSummary[1:top_n_features, ]
  }

  # make sure that both lower and upper bounds of the confidence interval are on the same side of zero
  if (nonzeroCI) {
    onedirection <- (rowSummary$lowerCI > 0 & rowSummary$upperCI > 0) | (rowSummary$lowerCI < 0 & rowSummary$upperCI < 0)
    rowSummary <- rowSummary[onedirection, ]
  }

  # PLOT
  # ============================================================
  ftr <- rowSummary$feature
  MEAN <- rowSummary$mean
  lci <- rowSummary$lowerCI
  uci <- rowSummary$upperCI

  Plot <- ggplot(data = NULL,
                 aes(x = ftr,
                     y = MEAN)) +
    geom_col(fill = "#07B86B", alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "solid", color = "gray70", alpha = 0.75, linewidth = 0.7) +
    geom_errorbar(aes(ymin = lci,
                      ymax = uci),
                  width = 0.2, color = "#7A004BF0",
                  alpha = 0.75, linewidth = 0.7) +

    coord_flip() +  # Rotating the graph to have mean values on X-axis
    ggtitle("") +
    xlab("Features\n") +
    ylab(paste("\nWeighted Mean SHAP contributions for row", row_index)) +
    theme_classic() +
    # Reduce top plot margin
    theme(plot.margin = margin(t = -0.5,
                               r = .25,
                               b = .25,
                               l = .25,
                               unit = "cm")) +
    # Set lower limit of expansion to 0
    scale_y_continuous(expand = expansion(mult = c(0.025, 0.025)))

  Plot$rowSummarySHAP <- rowSummary

  if (plot) print(Plot)
  if (print) print(rowSummary)

  return(list(summary = rowSummary,
              plot = Plot))

}
