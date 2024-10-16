#' @title Plot weighted SHAP contributions
#' @description This function applies different criteria to visualize SHAP contributions
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param plot character, specifying the type of the plot, which can be either
#'            'bar', 'waffle', or 'shap'. The default is 'bar'.
#' @param method character, specifying the method used for identifying the most
#'               important features according to their weighted SHAP values.
#'               The default selection method is "AUTO", which selects a method
#'               based on number of models that have been evaluated because
#'               lowerCI method is not applicable to SHAP values of a single
#'               model. If 'lowerCI' is specified,
#'               features whose lower weighted confidence interval exceeds the
#'               predefined 'cutoff' value would be reported.
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
#'               the top features. the cutoff value changes the features selection
#'               for all figures.
#' @param top_n_features integer. if specified, the top n features with the
#'                       highest weighted SHAP values will be selected, overrullung
#'                       the 'cutoff' and 'method' arguments.
#' @param features character vector, specifying the feature to be plotted.
#' @param domains character list, specifying the domains for grouping the features'
#'                contributions. Domains are clusters of features' names, that
#'                can be used to compute WMSHAP at higher level, along with
#'                their 95% confidence interval. This computation can be used to
#'                better understand how a cluster of features influence the
#'                outcome. Note that either of 'features' or 'domains' arguments
#'                can be specified at the time.
#' @param legendstyle character, specifying the style of the plot legend, which
#'                    can be either 'continuous' (default) or 'discrete'. the
#'                    continuous legend is only applicable to 'shap' plots and
#'                    other plots only use 'discrete' legend.
#' @param scale_colour_gradient character vector for specifying the color gradients
#'                              for the plot.
#' @param print logical. if TRUE, the WMSHAP summary table for the given row is printed
#' @importFrom waffle waffle
#' @importFrom h2o h2o.shap_summary_plot h2o.getModel
#' @importFrom ggplot2 scale_colour_gradient2 theme guides guide_legend guide_colourbar
#'             margin element_text theme_classic labs ylab xlab ggtitle
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


shapley.plot <- function(shapley,
                         plot = "bar",
                         method = "AUTO",
                         cutoff = 0.005,
                         top_n_features = NULL,
                         features = NULL,
                         domains = NULL,
                         legendstyle = "continuous",
                         scale_colour_gradient = NULL,
                         print = FALSE) {

  # Variable definitions
  # ============================================================
  mean      <- NA
  shapratio <- NA
  COLORCODE <- c("#07B86B", "#07a9b8","#b86207","#b8b207", "#b80786",
                 "#073fb8", "#b8073c", "#8007b8", "#bdbdbd", "#4eb807")

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (!is.character(plot)) {
    stop("plot must be a character string")
  }
  if (plot != "bar" & plot != "waffle" & plot != "shap") {
    stop("plot must be either 'bar', 'waffle', or 'shap'")
  }

  if (plot == "waffle" & !is.null(features)) stop("'features' are not available in 'waffle' plot")
  else if (plot == "waffle" & !is.null(domains)) stop("'domains' are not available in 'waffle' plot")

  if (!is.null(features) & !is.null(domains)) stop("either specify 'features' or 'domains'")

  if (method == "AUTO") {
    if (length(shapley[["ids"]]) > 1) method = "lowerCI"
    else if (length(shapley[["ids"]]) == 1) method = "mean"
    else stop("at least 1 valid model is needed")
  }

  # Feature selection
  # ============================================================
  if (length(shapley[["ids"]]) >= 1) {

    #??? shapley.feature.selection needs update because some of the values
    #    it returns were useless and buggy. In this version, those values were
    #    replaced, but in future updates, the function should be improved and
    #    properly tested
    select <- shapley.feature.selection(shapley = shapley,
                                        method = method,
                                        cutoff = cutoff,
                                        top_n_features = top_n_features,
                                        features = features)

    # select$shapley$summaryShaps$feature <- as.character(select$shapley$summaryShaps$feature)
    # SUMMARY   <- select$shapley$summaryShaps[select$shapley$summaryShaps$feature %in%
    #                                            select$features[!is.na(select$features)], ]
    SUMMARY   <- select$shapley$summaryShaps
    SUMMARY   <- SUMMARY[order(SUMMARY$mean, decreasing = TRUE), ]
    shapratio <- select$shapratio #[!is.na(select$features)]
  }
  else stop("no model ID was found")

  # Change the plot reflect on single SHAP vs. multimodel WMSHAP
  # ============================================================
  if (length(shapley[["ids"]]) > 1 & is.null(domains)) {
    FILLCOLOR <- "#07B86B"
    YLAB      <- "\nMean absolute SHAP contributions with 95% CI"
  }
  else if (length(shapley[["ids"]]) == 1 & is.null(domains)) {
    FILLCOLOR <- "#07a9b8"
    YLAB      <- "\nAbsolute SHAP contributions of features"
  }
  else if (length(shapley[["ids"]]) == 1 & !is.null(domains)) {
    FILLCOLOR <- "#b8b207"
    YLAB      <- "\nAbsolute SHAP contributions of domains"
  }
  else if (length(shapley[["ids"]]) > 1 & !is.null(domains)) {
    if (length(domains) <= 10) FILLCOLOR <- COLORCODE[1:length(domains)]
    else FILLCOLOR <- COLORCODE[1]
    YLAB      <- "\nAbsolute SHAP contributions of domains"
  }

  # ############################################################
  # DOMAINS ANALYSIS
  # ############################################################
  if (!is.null(domains)) {
    UNQ          <- names(domains)
    domainsShaps <- data.frame(
      domains = UNQ,
      mean = NA,
      sd = NA,
      ci = NA,
      lowerCI = NA,
      upperCI = NA)
  }

  # Print the bar plot at FEATURE level
  # ============================================================
  if (plot == "bar" & is.null(domains) ) {
    SUMMARY$feature <- factor(
      SUMMARY$feature,
      levels = SUMMARY$feature[order(
        SUMMARY[["mean"]])])

    ftr <- SUMMARY$feature #FEATURES
    #nrmm <- SUMMARY$mean
    lci <- SUMMARY$lowerCI
    uci <- SUMMARY$upperCI

    Plot <- ggplot(data = NULL,
                   aes(x = ftr,
                       y = SUMMARY$mean)) +
      geom_col(fill = FILLCOLOR, alpha = 0.8) +
      geom_errorbar(aes(ymin = lci,
                        ymax = uci),
                    width = 0.2, color = "#7A004BF0",
                    alpha = 0.75, linewidth = 0.7) +
      coord_flip() +  # Rotating the graph to have mean values on X-axis
      ggtitle("") +
      xlab("Features\n") +
      ylab(YLAB) +
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

  # Print the bar plot at DOMAIN level
  # ============================================================
  if (plot == "bar" & !is.null(domains) ) {
    w <- shapley$weights
    # get the raw data
    results <- as.data.frame(shapley$results)
    # Compute total absolute SHAP contributions per model
    TOTAL <- rowSums(abs(as.data.frame(shapley$feature_importance)), na.rm = TRUE)
    results <- sweep(results, 1, TOTAL, "/")

    SUMMARY <- data.frame(feature = names(domains),
                          mean = NA,
                          sd = NA,
                          ci = NA)

    for (i in 1:length(domains)) {
      getnames <- domains[[i]]
      tmp <- rowSums(results[, getnames])

      weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
      weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
      weighted_sd   <- sqrt(weighted_var)
      ci            <- 1.96 * weighted_sd / sqrt(length(tmp))

      SUMMARY[i, "mean"] <- weighted_mean
      SUMMARY[i, "sd"]   <- weighted_sd
      SUMMARY[i, "ci"]   <- ci

      # Compute the lower and upper confidence intervals
      # -------------------------------------------------------------
      SUMMARY[i, "lowerCI"] <- weighted_mean - ci
      SUMMARY[i, "upperCI"] <- weighted_mean + ci
    }

    SUMMARY$feature <- factor(
      SUMMARY$feature,
      levels = SUMMARY$feature[order(
        SUMMARY[["mean"]])])

    ftr <- SUMMARY$feature #FEATURES
    lci <- SUMMARY$lowerCI
    uci <- SUMMARY$upperCI


    Plot <- ggplot(data = NULL,
                   aes(x = ftr,
                       y = SUMMARY$mean)) +
      geom_col(fill = FILLCOLOR, alpha = 0.8) +
      geom_errorbar(aes(ymin = lci,
                        ymax = uci),
                    width = 0.2, color = "#7A004BF0",
                    alpha = 0.75, linewidth = 0.7) +
      coord_flip() +  # Rotating the graph to have mean values on X-axis
      ggtitle("") +
      xlab("Features\n") +
      ylab(YLAB) +
      theme_classic() +
      # Reduce top plot margin
      theme(plot.margin = margin(t = -0.5,
                                 r = .25,
                                 b = .25,
                                 l = .25,
                                 unit = "cm")) +
      # Set lower limit of expansion to 0
      scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

    Plot$domainSummary <- SUMMARY
  }

  # Waffle plot
  # ============================================================
  else if (plot == "waffle") {
    round_to_half <- function(x) {
      return(round(x * 2) / 2)
    }

    # Calculate the percentages
    # -------------------------------------------------------------
    percentage <- round(shapratio*100, 2)

    # use the value of the cutoff as smallest shapratio allowed
    shapratio <- round_to_half(shapratio*(1/cutoff))

    # Create a factor with the percentage for the legend
    legend <- paste0(SUMMARY$feature, " (", percentage, "%)")

    # Order the legend by descending percentage
    #legend <- factor(legend, levels = legend[order(-percentage)])
    names(shapratio) <- as.character(legend)


    colors <- NA
    if (length(shapratio) >= 9) {
      color <- grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
      colors <- sample(color, length(shapratio))
    }

    Plot <- waffle(shapratio, rows = 20, size = 1, colors = colors,
                   title = "Weighted mean SHAP contributions",
                   legend_pos = "right")
  }

  # SHAP Plot
  # ============================================================
  else if (plot == "shap") {
    # STEP 3: PLOT
    # ============================================================
    select$shapley$contributionPlot$data <- select$shapley$contributionPlot$data[
      select$shapley$contributionPlot$data$feature %in% SUMMARY$feature,
    ]

    Plot <- select$shapley$contributionPlot +
      ggtitle("") +
      xlab("Features\n") +
      ylab("\nSHAP contribution") +
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

  print(Plot)
  if (print) print(SUMMARY)

  return(Plot)
}

# print(shapley.plot(shapley = shapley, plot = "waffle", method = "mean",
#                    domains = list(Demographic = c("RACE", "AGE"),
#                                   Cancer = c("VOL", "PSA", "GLEASON"),
#                                   Tests = c("DPROS", "DCAPS")),
#                    print = TRUE
# ))



# print(shapley.plot(shapley, plot = "waffle", method = "shapratio", print = F,
#                    cutoff = 0.005
#                    #, features = c("PSA", "AGE")
#                    ))

# print(shapley.plot(shapley, plot = "waffle", method = "shapratio", print = F,
#                    cutoff = 0.01
#                    #, features = c("PSA", "AGE")
# ))

# print(shapley.plot(shapley, plot = "shap", method = "shapratio", print = F,
#                    cutoff = 0.01
#                    #, features = c("PSA", "AGE")
# ))

# print(shapley.plot(shapley = globalshap, plot = "bar", method = "shapratio",
#                    features = NULL#c("PSA", "AGE")
# ))
