#' @title Plot weighted mean SHAP (WMSHAP) contributions
#' @description Visualizes WMSHAP summaries from a \code{shapley} object. Features can be selected
#'              using \code{method} and \code{method/cutoff}, \code{top_n_features},
#'              or explicit \code{features} to specify feature selection method.
#' @param shapley object of class \code{"shapley"}, as returned by the 'shapley' function
#' @param plot Character. One of \code{"bar"} or \code{"wmshap"}.
#' @param method Character. One of \code{"mean"} or \code{"lowerCI"}; used by
#'               \code{feature.selection()} for feature selection
#'               when \code{top_n_features} or \code{features} are not set.
#' @param cutoff Numeric cutoff for \code{method} selection.
#' @param top_n_features Integer. If set, selects top N features by WMSHAP
#'                       (overrides cutoff and method arguments).
#' @param features Character vector, specifying the feature to be plotted
#'                 (overrides cutoff and method arguments).
#' @param legendstyle Character. For \code{plot = "wmshap"} only: \code{"continuous"} (default)
#'                    or \code{"discrete"}.
#' @param scale_colour_gradient Optional character vector of length 3, specifying
#'                              color names: \code{c(low, mid, high)}.
#'                              Used only when \code{plot = "wmshap"}.
#' @param labels Optional named character vector mapping feature names to display labels.
#'               To specify the labels, use the \code{c} function and for each feature,
#'               provide a label. For example, \code{c(feature1 = label2, feature2 = label2, ...)}.
# @importFrom waffle waffle
#' @importFrom h2o h2o.shap_summary_plot h2o.getModel
#' @importFrom ggplot2 scale_colour_gradient2 theme guides guide_legend guide_colourbar
#'             margin element_text theme_classic labs ylab xlab ggtitle
#'             coord_flip scale_y_continuous scale_x_discrete
#' @author E. F. Haghish
#' @return A ggplot object
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
#' shapley.plot(result, plot = "wmshap")
#' }
#' @export

shapley.plot <- function(shapley,
                         plot = "bar",
                         method = "mean",
                         cutoff = 0.01,
                         top_n_features = NULL,
                         features = NULL,
                         legendstyle = "continuous",
                         scale_colour_gradient = NULL,
                         labels = NULL) {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (!is.character(plot)) {
    stop("plot must be a character string")
  }
  # for syntax change on version 0.6
  if (plot == "shap") {
    plot <- "wmshap"
    warning("plot = 'shap' is deprecated; use plot = 'wmshap'.", call. = FALSE)
  }
  if (plot != "bar" & plot != "wmshap") {
    stop("plot must be either 'bar' or 'wmshap'")
  }

  # Feature selection
  # ============================================================
  select <- feature.selection(shapley = shapley,
                              method = method,
                              cutoff = cutoff,
                              top_n_features = top_n_features,
                              features = features)

  shapley   <- select$shapley                    # update the data for different plots
  features  <- select$features
  mean      <- select$mean
  shapratio <- select$mean #also mean


  # Print the bar plot
  # ============================================================
  if (plot == "bar") {
    shapley$summaryShaps$feature <- factor(
      shapley$summaryShaps$feature,
      levels = shapley$summaryShaps$feature[order(
        shapley$summaryShaps[["mean"]])])
    #summaryShaps <<- summaryShaps
    ftr <- shapley$summaryShaps$feature
    nrmm <- shapley$summaryShaps$mean
    lci <- shapley$summaryShaps$lowerCI
    uci <- shapley$summaryShaps$upperCI

    Plot <- ggplot(data = NULL,
                   aes(x = ftr,
                       y = nrmm)) +
      geom_col(fill = "#07B86B", alpha = 0.8) +
      geom_errorbar(aes(ymin = lci,
                        ymax = uci),
                    width = 0.2, color = "#7A004BF0",
                    alpha = 0.75, linewidth = 0.7) +
      coord_flip() +  # Rotating the graph to have mean values on X-axis
      ggtitle("") +
      xlab("Features\n") +
      ylab("\nWeighted Mean SHAP Ratio with 95% CI") +
      theme_classic() +
      # Reduce top plot margin
      theme(plot.margin = margin(t = -0.5,
                                 r = .25,
                                 b = .25,
                                 l = .25,
                                 unit = "cm")) +
      # Set lower limit of expansion to 0
      scale_y_continuous(expand = expansion(mult = c(0, 0.05)))
  }

  # else if (plot == "waffle") {
  #
  #   # round_to_half <- function(x) {
  #   #   return(round(x * 2) / 2)
  #   # }
  #   #
  #   # # Calculate the percentages
  #   # percentage <- round((mean / sum(mean) * 100), 2)
  #   #
  #   # shapratio <- round_to_half(shapratio*400)
  #   #
  #   # # Create a factor with the percentage for the legend
  #   # legend <- paste0(features, " (", percentage, "%)")
  #   # # Order the legend by descending percentage
  #   # #legend <- factor(legend, levels = legend[order(-percentage)])
  #   # names(shapratio) <- as.character(legend)
  #   #
  #   # colors <- NA
  #   # if (length(shapratio) >= 9) {
  #   #   color <- grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
  #   #   colors <- sample(color, length(shapratio))
  #   # }
  #   #
  #   # Plot <- waffle(shapratio, rows = 20, size = 1, colors = colors,
  #   #                title = "Weighted mean SHAP contributions",
  #   #                legend_pos = "right")
  #   stop("due to an issue with the 'waffle' package on CRAN, this function is temporarily disabbled")
  # }

  else if (plot == "wmshap") {

    # Make sure the features are in the correct order
    # ============================================================
    shapley$contributionPlot$data$feature <- factor(
      shapley$summaryShaps$feature,
      levels = shapley$summaryShaps$feature[order(
        shapley$summaryShaps[["mean"]])])

    # STEP 3: PLOT
    # ============================================================
    Plot <- shapley$contributionPlot +
      ggtitle("") +
      xlab("Features\n") +
      ylab("\nWeighted Mean SHAP contributions") +
      theme_classic() +
      labs(colour = "Normalized values") +
      theme(
        legend.position="top",
        legend.justification = "right",
        legend.title.align = 0.5,
        legend.direction = "horizontal",
        legend.text=element_text(colour="black", size=6, face="bold"),
        legend.key.height = grid::unit(0.3, "cm"),
        legend.key.width = grid::unit(1, "cm"),
        #legend.margin=margin(grid::unit(0,"cm")),
        legend.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "cm"),
        plot.margin = margin(t = -0.5, r = .25, b = .25, l = .25, unit = "cm")  # Reduce top plot margin
      ) +
      guides(colour = guide_colourbar(title.position = "top", title.hjust = 0.5))

    if (legendstyle == "continuous") {
      # Set color range
    }

    else if (legendstyle == "discrete") {
      Plot <- Plot +
        guides(colour = guide_legend(title.position = "top",
                                     title.hjust = 0.5,
                                     legend.margin = margin(t = -1, unit = "cm"),
                                     override.aes = list(size = 3)
        )) +
        theme(legend.key.height = grid::unit(0.4, "cm"),
              legend.key.width = grid::unit(0.4, "cm"))
    }

    # Fix the color scale of the model
    # ============================================================
    if (length(scale_colour_gradient) == 3) {
      Plot <- Plot +
        scale_colour_gradient2(low=scale_colour_gradient[1],
                               mid=scale_colour_gradient[2],
                               high=scale_colour_gradient[3],
                               midpoint = 0.5)
    }
  }
  else {
    stop("plot must be either 'bar' or 'wmshap'")
  }

  # Add the labels for features, if specified
  # ============================================================
  if (!is.null(Plot) & !is.null(labels)) {
    Plot <- Plot + scale_x_discrete(labels = labels)
  }

  print(Plot)
  return(Plot)
}


