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


shapley.plot <- function(shapley, domains,
                         plot = "bar",
                         method = "AUTO",
                         legendstyle = "continuous",
                         scale_colour_gradient = NULL,
                         print = FALSE) {

  # Variable definitions
  # ============================================================
  DOMAINS <- names(domains)
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

  # Feature selection
  # ============================================================
  if (length(shapley[["ids"]]) < 1) stop("no model ID was found")

  # Change the plot reflect on single SHAP vs. multimodel WMSHAP
  # ============================================================
  if (length(shapley[["ids"]]) > 1 ) {
    if (length(domains) <= 10) FILLCOLOR <- COLORCODE[1:length(domains)]
    else FILLCOLOR <- COLORCODE[1]
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



  # Print the bar plot at DOMAIN level
  # ============================================================
  if (plot == "bar" & !is.null(domains) ) {
    w <- shapley$weights
    # create a raw dataset for domain data
    results <- as.data.frame(shapley$results)
    results$feature <- as.character(results$feature)
    results$Row.names <- NA
    names(results)[1] <- "domain"

    # add the domain names based on specified features
    for (i in 1:length(domains)) {
      getnames <- domains[[i]]

      # make sure the names are correct
      for (j in getnames) {
        if (!(j %in% results$feature)) stop(paste(j, "was not in the dataframe"))
        results$domain[results$feature == j] <- names(domains)[i]
      }
    }
    results$feature <- NULL
    results <- na.omit(results)
    #View(na.omit(results))

    # aggregate
    aggregate_contribution <- function(df, column) {
      aggregate(formula(paste(column, "~ domain + index")), data = results, sum)
    }

    # aggregate domain data at row level
    # ============================================================
    domainRow <- NULL
    for (i in grep("^contribution", names(results))) {
      aggregated_df <- aggregate_contribution(results, names(results)[i])
      if (is.null(domainRow)) {
        domainRow <- aggregated_df
      } else {
        domainRow <- merge(domainRow, aggregated_df, by = c("domain", "index"))
      }
    }

    # aggregate domain data at column level
    # ============================================================
    contributionNames <- names(domainRow)[grep("^contribution", names(domainRow))]
    absSum <- function(x) return(sum(abs(x)))
    aggregate_column <- function(df, column) {
      aggregate(formula(paste(column, "~ domain")), data = domainRow, absSum)
    }
    domainColumn <- NULL
    for (i in contributionNames) {
      aggregated_df <- aggregate_column(domainRow, i)
      if (is.null(domainColumn)) {
        domainColumn <- aggregated_df
      }
      else {
        domainColumn <- cbind(domainColumn, aggregated_df[,2])
      }
    }
    colnames(domainColumn) <- c("domain", contributionNames)

    # Compute domains' ratios of contributions at column level
    # ============================================================
    domainRatio <- domainColumn
    TOTAL <- colSums(domainColumn[, contributionNames], na.rm = TRUE)
    domainRatio[, contributionNames] <- sweep(domainRatio[, contributionNames], 2, TOTAL, "/")

    # # Compute relative contributions at column level
    # # ============================================================
    # domainRelative <- domainColumn
    # MAX <- apply(domainRelative[, contributionNames], 2, max)
    # domainRelative[, contributionNames] <- sweep(domainRelative[, contributionNames], 2, MAX, "/")
    # View(domainRelative)

    # create a summary dataset to store cross-model domain contributions
    # ----------------------------------------------------------
    SUMMARY <- data.frame(domain = names(domains),
                          mean = NA,
                          sd = NA,
                          ci = NA,
                          lowerCI = NA,
                          upperCI = NA)

    ### COMPUTING FOR RATIO VALUES
    for (i in unique(domainRatio$domain)) {
      tmp <- domainRatio[domainRatio$domain == i, contributionNames]
      weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
      weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
      weighted_sd   <- sqrt(weighted_var)
      ci            <- 1.96 * weighted_sd / sqrt(length(tmp))
      SUMMARY[SUMMARY$domain == i, "mean"] <- weighted_mean
      SUMMARY[SUMMARY$domain == i, "sd"]   <- weighted_sd
      SUMMARY[SUMMARY$domain == i, "ci"]   <- ci

      # Compute the lower and upper confidence intervals
      # -------------------------------------------------------------
      SUMMARY[SUMMARY$domain == i, "lowerCI"] <- weighted_mean - ci
      SUMMARY[SUMMARY$domain == i, "upperCI"] <- weighted_mean + ci
    }

    ### COMPUTING FOR RELATIVE VALUES
    # for (i in unique(domainRelative$domain)) {
    #   tmp <- domainRelative[domainRelative$domain == i, contributionNames]
    #   weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
    #   weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
    #   weighted_sd   <- sqrt(weighted_var)
    #   ci            <- 1.96 * weighted_sd / sqrt(length(tmp))
    #   print(i)
    #   SUMMARY[SUMMARY$domain == i, "mean"] <- weighted_mean
    #   SUMMARY[SUMMARY$domain == i, "sd"]   <- weighted_sd
    #   SUMMARY[SUMMARY$domain == i, "ci"]   <- ci
    #
    #   # Compute the lower and upper confidence intervals
    #   # -------------------------------------------------------------
    #   SUMMARY[SUMMARY$domain == i, "lowerCI"] <- weighted_mean - ci
    #   SUMMARY[SUMMARY$domain == i, "upperCI"] <- weighted_mean + ci
    # }

    SUMMARY$domain <- factor(
      SUMMARY$domain,
      levels = SUMMARY$domain[order(
        SUMMARY[["mean"]])])

    ftr <- SUMMARY$domain #FEATURES
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
      xlab("Domains\n") +
      ylab("\nWeighted Mean SHAP contributions of domains") +
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
    Plot$domainRatio <- domainRatio
  }

  print(Plot)
  if (print) print(SUMMARY)

  return(Plot)
}

# print(shapley.plot(shapley = shapley, plot = "bar", method = "mean",
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
