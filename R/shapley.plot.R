#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param plot character, specifying name of the plot. the default is 'bar',
#'             creating a barplot of the mean weighted feature importance alongside
#'             their confidence intervals. Other options are "pie", "tree", and
#'             "waffle".
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
#'             geom_bar coord_polar theme_void element_blank element_text
#'             scale_fill_discrete labs
#' @importFrom treemapify geom_treemap geom_treemap_text
#' @importFrom waffle waffle
#' @author E. F. Haghish
#' @return ggplot2 plot
#' @examples
#' \dontrun{
#' # load the required libraries for building the base-learners and the ensemble models
#' library(h2o)            #shapley supports h2o models
#' library(autoEnsemble)   #autoEnsemble models, particularly useful under severe class imbalance
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
#' shapley.plot(result, plot = "pie")
#' shapley.plot(result, plot = "tree")
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
  if (plot != "bar" & plot != "pie" & plot != "tree" & plot != "waffle") {
    stop("plot must be either 'bar', 'pie', 'tree', or 'waffle'")
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

  # Pie chart
  # ============================================================
  if (plot == "pie") {
    # Create the pie chart
    Plot <- ggplot(data=NULL, aes(x = "",
                                  y = normalized_mean, fill = legend)) +
      geom_bar(width = 1, stat = "identity") +
      coord_polar("y", start = 0) +
      theme_void() +
      theme(
        legend.title = element_blank(),
        legend.position = "right",
        text = element_text(size = 12) # Adjust text size here if needed
      ) +
      scale_fill_discrete(name = "Feature", labels = legend) +
      labs(fill = "Feature")
  }

  else if (plot == "tree") {
    Plot <- ggplot(data=NULL,
                   aes(area = normalized_mean,
                       fill = features,
                       label = features)) +  #legend
      geom_treemap() +
      geom_treemap_text(
        #colour = "white",
        place = "centre",
        grow = TRUE, reflow = TRUE, size = 1) + # Fixed font size instead of letting it be determized automatically
      theme(legend.position = "right") +
      labs(fill = "Feature") +
      scale_fill_discrete(name = "Feature", labels = legend)
  }
  else if (plot == "waffle") {
    names(shapratio) <- as.character(legend)
    Plot <- waffle(shapratio, rows = 20, size = 1,
                          title = "Weighted mean SHAP contributions",
                   legend_pos = "right")
  }

  print(Plot)

  return(Plot)
}


