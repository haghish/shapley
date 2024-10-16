#' @title Weighted mean SHAP values computed at subject level
#' @description Weighted mean of SHAP values and weighted SHAP confidence intervals
#'              provide a measure of feature importance for a grid of fine-tuned models
#'              or base-learners of a stacked ensemble model at subject level,
#'              showing that how each feature influences the prediction made for
#'              a row in the dataset and to what extend different models agree
#'              on that effect. If the 95% confidence interval crosses the
#'              vertical line at 0.00, then it can be concluded that the feature
#'              does not significantly influences the subject, when variability
#'              across models is taken into consideration.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param row_index subject or row number in a wide-format dataset to be visualized
#' @param features character vector, specifying the feature to be plotted.
#' @param plot logical. if TRUE, the plot is visualized.
#' @param print logical. if TRUE, the WMSHAP summary table for the given row is printed
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
#' @author E. F. Haghish
#' @return a list including the GGPLOT2 object, the data frame of SHAP values,
#'         and performance metric of all models, as well as the model IDs.
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
#' ### PREPARE AutoML Grid (takes a couple of minutes)
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
#' #######################################################
#' ### PREPARE H2O Grid (takes a couple of minutes)
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
#' #######################################################
#' ### PREPARE autoEnsemble STACKED ENSEMBLE MODEL
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
#' shapley.row.plot(shapley, row_index = 11)
#'
#' #plot only the given features
#' shapPlot <- shapley.row.plot(shapley, row_index = 11, features = c("PSA", "AGE"))
#'
#' # inspect the computed data for the row 11
#' ptint(shapPlot$rowSummarySHAP)
#' }
#' @export

shapley.row.plot <- function(shapley,
                             row_index,
                             features = NULL,
                             plot = TRUE,
                             print = FALSE
) {

  # Variable definitions
  # ============================================================
  w         <- shapley$weights
  # COLORCODE <- c("#07B86B", "#07a9b8","#b86207","#b8b207", "#b80786",
  #                "#073fb8", "#b8073c", "#8007b8", "#bdbdbd", "#4eb807")

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

  #index   <- substr(results$Row.names, 1, nchar(rowname)) == rowname
  results <- results[results$index == row_index, ]

  # compute WMSHAP for the row
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

  ftr <- rowSummary$feature #FEATURES
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

}

# shapley.row.plot(shapley, row_index = 11, features = c("PSA", "AGE"))
