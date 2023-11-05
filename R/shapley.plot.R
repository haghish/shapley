#' @title Plot weighted SHAP contributions
#' @description This function applies different criteria to visualize SHAP contributions
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param plot character, specifying the type of the plot, which can be either
#'            'bar' or 'waffle'. The default is 'bar'
#' @importFrom waffle waffle
#' @author E. F. Haghish
#' @return ggplot object
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
#' ### PLOT THE WEIGHTED MEAN SHAP VALUES
#' #######################################################
#'
#' shapley.plot(result, plot = "bar")
#' shapley.plot(result, plot = "waffle")
#' }
#' @export


shapley.plot <- function(shapley, plot = "bar") {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (!is.character(plot)) {
    stop("plot must be a character string")
  }
  if (plot != "bar" & plot != "waffle") {
    stop("plot must be either 'bar' or 'waffle'")
  }

  index <- order(- shapley$summaryShaps$normalized_mean)
  features <- shapley$summaryShaps$feature[index]
  normalized_mean <- shapley$summaryShaps$normalized_mean[index]
  shapratio <- shapley$summaryShaps$shapratio[index]

  # Print the bar plot
  # ============================================================
  if (plot == "bar") {
    Plot <- shapley$plot
  }

  # Calculate the percentages
  percentage <- round((normalized_mean / sum(normalized_mean) * 100), 2)

  round_to_half <- function(x) {
    return(round(x * 2) / 2)
  }
  shapratio <- round_to_half(shapratio*400)

  # Create a factor with the percentage for the legend
  legend <- paste0(features, " (", percentage, "%)")
  # Order the legend by descending percentage
  #legend <- factor(legend, levels = legend[order(-percentage)])

  if (plot == "waffle") {
    names(shapratio) <- as.character(legend)
    Plot <- waffle(shapratio, rows = 20, size = 1,
                   title = "Weighted mean SHAP contributions",
                   legend_pos = "right")
  }

  print(Plot)

  return(Plot)
}

