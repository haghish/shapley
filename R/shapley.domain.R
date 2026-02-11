# ??? The running speed of this function can be improved

#' @title Compute and plot weighted mean SHAP contributions at group level (factors or domains)
#' @description Aggregates SHAP contributions across user-defined domains (groups of features),
#'              computes weighted mean and an 95% CI across models, and
#'              returns a plot plus summary tables.
#' @param shapley Object of class \code{"shapley"}, as returned by the 'shapley' function
#' @param domains Named list of character vectors. Each element name is a domain name;
#'                each element value is a character vector of feature names assigned to that domain.
#' @param plot Logical. If \code{TRUE}, a bar plot of domain WMSHAP contributions is created.
#' @param colorcode Character vector for specifying the color names for each domain in the plot.
#' @param print Logical. If TRUE, prints the domain WMSHAP summary table.
#' @importFrom stats na.omit aggregate formula
#' @importFrom h2o h2o.shap_summary_plot h2o.getModel
#' @importFrom ggplot2 scale_colour_gradient2 theme guides guide_legend guide_colourbar
#'             margin element_text theme_classic labs ylab xlab ggtitle
#' @author E. F. Haghish
#' @return A list with:
#' \describe{
#'   \item{domainSummary}{Data frame with WMSHAP domain contributions and CI.}
#'   \item{domainRatio}{Data frame with per-model WMSHAP domain contribution ratios.}
#'   \item{plot}{A ggplot object (or NULL if plotting not requested/implemented).}
#' }
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
#'
#' #######################################################
#' ### DEFINE DOMAINS (GROUPS OF FEATURES OR FACTORS)
#' #######################################################
#' shapley.domain(shapley = result, plot = TRUE,
#'                domains = list(Demographic = c("RACE", "AGE"),
#'                               Cancer = c("VOL", "PSA", "GLEASON"),
#'                               Tests = c("DPROS", "DCAPS")),
#'                               print = TRUE
# ))
#' }
#' @export

shapley.domain <- function(shapley,
                           domains,
                           plot = TRUE,
                           colorcode = NULL,
                           print = FALSE,
                           xlab = "Domains") {

  # Variable definitions
  # ============================================================
  DOMAINS <- names(domains)
  COLORCODE <- c("#855C75FF", "#D9AF6BFF", "#AF6458FF","#736F4CFF","#526A83FF", "#625377FF", "#68855CFF", "#9C9C5EFF", "#A06177FF", "#8C785DFF", "#467378FF", "#7C7C7CFF")
  FILLCOLOR <- NULL
  mean      <- NA
  Plot      <- NULL

  if (!is.null(colorcode)) COLORCODE <- colorcode

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
      aggregate(formula(paste(column, "~ domain + index")), data = df, sum)
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
  }


  # Bar plot at DOMAIN level
  # ============================================================
  if (plot & !is.null(domains) ) {

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
      xlab(paste0(xlab,"\n")) +
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
  }

  if (plot) print(Plot)
  if (print) print(SUMMARY)

  return(list(domainSummary = SUMMARY,
              domainRatio   = domainRatio,
              plot          = Plot))
}


