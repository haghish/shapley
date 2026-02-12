#' @title Flag and rank features by WMSHAP cutoffs
#' @description This function applies different criteria simultaniously to identify
#'              the most important features in a model. The criteria include:
#'              1) minimum limit of lower weighted confidence intervals of SHAP values
#'              relative to the feature with highest SHAP value.
#'              2) minimum limit of percentage of weighted mean SHAP values relative to
#'              over all SHAP values of all features. These are specified with two
#'              different cutoff values.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param mean Numeric. specifying the cutoff of weighted mean
#'                         SHAP ratio (WMSHAP). The default is 0.01. Lower values will
#'                         be more generous in defining "importance", while higher values
#'                         are more restrictive. However, these default values are not
#'                         generalizable to all situations and algorithms.
#' @param lowerCI numeric. Specifying the limit of lower bound of 95\% WMSHAP
#'                         The default is 0.01. Lower values will
#'                         be more generous in defining "importance", while higher values
#'                         are more restrictive. However, these default values are not
#'                         generalizable to all situations and algorithms.
#' @author E. F. Haghish
#' @return data.frame of selected features
#' @examples
#'
#' \dontrun{
#' # load the required libraries for building the base-learners and the ensemble models
#' library(h2o)            #shapley supports h2o models
#' library(shapley)
#'
#' # initiate the h2o server
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)
#'
#' # upload data to h2o cloud
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#'
#' ### H2O provides 2 types of grid search for tuning the models, which are
#' ### AutoML and Grid. Below, I demonstrate how weighted mean shapley values
#' ### can be computed for both types.
#'
#' set.seed(10)
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
#'                  # this setting ensures the models are comparable for building a meta learner
#'                  seed = 2023, nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' ### call 'shapley' function to compute the weighted mean and weighted confidence intervals
#' ### of SHAP values across all trained models.
#' ### Note that the 'newdata' should be the testing dataset!
#' result <- shapley(models = aml, newdata = prostate, plot = TRUE)
#'
#' #######################################################
#' ### Select top features
#' #######################################################
#' shapley.top(result, mean = 0.005, lowerCI = 0.01)
#' }
#' @export

shapley.top <- function(shapley, mean = 0.01, lowerCI = 0.01) {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")

  # Prepare the dataset
  # ============================================================
  results <- data.frame(
    feature = shapley$summaryShaps$feature,
    mean = shapley$summaryShaps$mean,
    lowerCI = shapley$summaryShaps$lowerCI
  )

  # evaluate the criteria
  # ============================================================
  results$mean_criteria <- results$mean >= mean
  results$lowerCI_criteria <- results$lowerCI >= lowerCI

  # Sort the results
  # ============================================================
  results <- results[order(results$mean_criteria & results$lowerCI_criteria,
                           decreasing = TRUE), ]

  return(results)
}


