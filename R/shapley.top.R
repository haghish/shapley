#' @title Select top features in a model
#' @description This function applies different criteria simultaniously to identify
#'              the most important features in a model. The criteria include:
#'              1) minimum limit of lower weighted confidence intervals of SHAP values
#'              relative to the feature with highest SHAP value.
#'              2) minimum limit of percentage of weighted mean SHAP values relative to
#'              over all SHAP values of all features. These are specified with two
#'              different cutoff values.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param lowerci numeric, specifying the lower limit of weighted confidence intervals
#'                     of SHAP values relative to the feature with highest SHAP value.
#'                     the default is 0.01
#' @param shapratio numeric, specifying the lower limit of percentage of weighted mean
#'                         SHAP values relative to over all SHAP values of all features.
#'                         the default is 0.005
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
#' ### Significance testing of contributions of two features
#' #######################################################
#'
#' shapley.top(result, lowerci = 0.01, shapratio = 0.005)
#' }
#' @export

shapley.top <- function(shapley, lowerci = 0.01, shapratio = 0.005) {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")

  # Prepare the dataset
  # ============================================================
  results <- data.frame(
    feature = shapley$summaryShaps$feature,
    lowerci = shapley$summaryShaps$lowerCI,
    shapratio = shapley$summaryShaps$shapratio
  )

  # evaluate the criteria
  # ============================================================
  results$lowerCI_criteria <- results$lowerci >= lowerci
  results$shapratio_criteria <- results$shapratio >= shapratio

  # Sort the results
  # ============================================================
  results <- results[order(results$lowerCI_criteria &
                           results$shapratio_criteria,
                           decreasing = TRUE), ]

  return(results)
}


