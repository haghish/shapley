#' @title Weighted average of SHAP values and weighted SHAP confidence intervals
#'        for a grid of fine-tuned models or base-learners of a stacked ensemble
#'        model
#' @description Weighted average of SHAP values and weighted SHAP confidence intervals
#'              provide a measure of feature importance for a grid of fine-tuned models
#'              or base-learners of a stacked ensemble model. Instead of reporting
#'              relative SHAP contributions for a single model, this function
#'              takes the variability in feature importance of multiple models
#'              into account and computes weighted mean and confidence intervals
#'              for each feature, taking the performance metric of each model as
#'              the weight. The function also provides a plot of the weighted
#'              SHAP values and confidence intervals. Currently only models
#'              trained by h2o machine learning software or autoEnsemble
#'              package are supported.
#' @param models H2O search grid, AutoML grid, a character vector of H2O model IDs,
#'               or an autoEnsemble object. the \code{"h2o.get_ids"} function
#'               from \code{"h2otools"} can
#'               retrieve the IDs from grids.
#' @param newdata h2o frame (data.frame). the data.frame must be already uploaded
#'                on h2o server (cloud). when specified, this dataset will be used
#'                for evaluating the models. if not specified, model performance
#'                on the training dataset will be reported.
#' @param performance_metric character, specifying the performance metric to be
#'                           used for weighting the SHAP values (mean and 95% CI). The default is
#'                           "aucpr" (area under the precision-recall curve). Other options include
#'                           "auc" (area under the ROC curve), "mcc" (Matthews correlation coefficient),
#'                           and "f2" (F2 score).
#' @param method character, specifying the method used for identifying the most
#'               important features according to their weighted SHAP values.
#'               The default selection method is "lowerCI", which includes
#'               features whose lower weighted confidence interval exceeds the
#'               predefined 'cutoff' value (default is relative SHAP of 1%).
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
#'               the top features.
#' @param top_n_features integer. if specified, the top n features with the
#'                       highest weighted SHAP values will be selected, overrullung
#'                       the 'cutoff' and 'method' arguments.
#' @param family character. currently only "binary" classification models trained
#'               by h2o machine learning are supported.
#' @param plot logical. if TRUE, the weighted mean and confidence intervals of
#'             the SHAP values are plotted. The default is TRUE.
#' @param sample_size integer. The number of observations to be sampled from the
#'                    data set. The default is all observations provided within
#'                    the newdata.
#' @param normalize_to character. The default value is 'feature', specifying that
#'                     in the normalization of the SHAP values, the maximum confidence
#'                     interval of the weighted SHAP values should be equal to
#'                     "1", in order to limit the plot values to maximum of one.
#'                     the alternative is "ci", which sets the feature with
#'                     the maximum SHAP value to one, allowing the higher CI to
#'                     go beyond one. Setting this value is mainly for aesthetic
#'                     reason to adjust the Plot, but also, it can influence the
#'                     feature selection process, depending on the method in use,
#'                     because it changes how the SHAP values should be normalized.
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
#' @importFrom h2o h2o.stackedEnsemble h2o.getModel h2o.auc h2o.aucpr h2o.mcc
#'             h2o.F2 h2o.mean_per_class_error h2o.giniCoef h2o.accuracy
#'             h2o.shap_summary_plot
#' @importFrom h2otools h2o.get_ids
#' @importFrom autoEnsemble autoEnsemble
#' @importFrom curl curl
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
#' @author E. F. Haghish
#' @export
#' @return a list including the GGPLOT2 object, the data frame of SHAP values,
#'         and performance metric of all models, as well as the model IDs.
importance <- function(models,
                       newdata,
                       plot = TRUE,
                       family = "binary",
                       performance_metric = c("aucpr"),
                       method = c("lowerCI"),
                       cutoff = 0.05,
                       top_n_features = NULL,
                       sample_size = row(newdata),
                       normalize_to = "ci") {

  # Variables
  # ============================================================
  w <- NULL
  results <- NULL
  feature_importance <- list()
  z <- 0
  pb <- txtProgressBar(z, length(ids), style = 3)

  # Syntax check
  # ============================================================
  if (family != "binary") stop("currently only binary classification models from 'h2o' and 'autoEnsemble' are supported")
  if (performance_metric != "aucpr" &
      performance_metric != "auc" &
      performance_metric != "mcc" &
      performance_metric != "f2") stop("performance metric must be one of 'aucpr', 'auc', 'mcc', or 'f2'")

  # STEP 0: prepare the models, by either confirming that the models are 'h2o' or 'autoEnsemble'
  #        or by extracting the model IDs from these objects
  # ============================================================
  if (inherits(models,"H2OAutoML") | inherits(models,"H2OAutoML")) {
    ids <- h2o.get_ids(models)
  }
  else if (inherits(models,"character")) {
    ids <- models
  }

  # STEP 1: Compute SHAP values and performance metric for each model
  # ============================================================
  for (i in ids) {
    z <- z + 1
    model <- h2o.getModel(i)

    m <- h2o.shap_summary_plot(
      model = model,
      newdata = newdata,
      columns = NULL, #get SHAP for all columns
      sample_size = sample_size
    )

    # Extract the performance metrics
    # ----------------------------------------------------------
    if (performance_metric == "aucpr") w <- c(w, h2o.aucpr(model))
    else if (performance_metric == "auc") w <- c(w, h2o.auc(model))
    else if (performance_metric == "mcc") w <- c(w, h2o.mcc(model))
    else if (performance_metric == "f2") w <- c(w, h2o.F2(model))

    # create the summary dataset
    # ----------------------------------------------------------
    if (z == 1) {
      data <- m$data #reserve the first model's data
      data <- data[order(data$Row.names), ]
      results <- data[, c("Row.names", "feature", "contribution")]
      BASE <- m
    }
    else {
      holder <- m$data[, c("Row.names", "contribution")]
      colnames(holder) <- c("Row.names", paste0("contribution", z))
      holder <- holder[order(holder$Row.names), ]
      results <- cbind(results, holder[, 2, drop = FALSE])
    }

    setTxtProgressBar(pb, z)
  }

  #data <<- cbind(data, results[, -1])

  # STEP 2: Calculate the summary shap values for each feature and store the mean
  #         shap values in a list, for significance testing
  # ============================================================
  summaryShaps <- data.frame(
    feature = unique(results$feature),
    mean = NA,
    sd = NA,
    ci = NA,
    normalized_mean = NA,
    normalized_ci = NA)

  globalVariables(c("feature", "mean", "sd", "ci", "normalized_mean", "normalized_ci"))

  for (j in unique(results$feature)) {
    tmp <- results[results$feature == j, grep("^contribution", names(results))]
    tmp <- colSums(abs(tmp))
    feature_importance[[j]] <- tmp

    weighted_mean <- weighted.mean(tmp, w)
    weighted_var  <- sum(w * (tmp - weighted_mean)^2) / (sum(w) - 1)
    weighted_sd   <- sqrt(weighted_var)

    # update the summaryShaps data frame
    summaryShaps[summaryShaps$feature == j, "mean"] <- weighted_mean #mean(tmp)
    summaryShaps[summaryShaps$feature == j, "sd"] <- weighted_sd
    summaryShaps[summaryShaps$feature == j, "ci"] <- 1.96 * weighted_sd / sqrt(length(tmp))
  }

  # STEP 3: NORMALIZE the SHAP contributions and their CI
  # ============================================================
  # the minimum contribution should not be normalized as zero. instead,
  # it should be the ratio of minimum value to the maximum value.
  # The maximum would be the highest mean + the highest CI
  if (normalize_to == "ci") {
    max  <- max(summaryShaps$mean + summaryShaps$ci)
  }
  else {
    max  <- max(summaryShaps$mean)
  }
  min  <- min(summaryShaps$mean)/max

  summaryShaps$normalized_mean <- normalize(x = summaryShaps$mean,
                                                min = min,
                                                max = max)

  summaryShaps$normalized_ci <- normalize(x = summaryShaps$ci,
                                          min = min,
                                          max = max)
  # compute relative shap values
  summaryShaps$shapratio <- summaryShaps$normalized_mean / sum(summaryShaps$normalized_mean)

  # compute lowerCI
  summaryShaps$lowerCI <- summaryShaps$normalized_mean - summaryShaps$normalized_ci
  summaryShaps$upperCI <- summaryShaps$normalized_mean + summaryShaps$normalized_ci

  # STEP 4: Feature selection
  # ============================================================
  if (!is.null(top_n_features)) {
    summaryShaps <- summaryShaps[order(summaryShaps$normalized_mean, decreasing = TRUE), ]
    summaryShaps <- summaryShaps[1:top_n_features, ]
  }
  else {
    if (method == "mean") {
      summaryShaps <- summaryShaps[summaryShaps$normalized_mean > cutoff, ]
    }
    else if (method == "shapratio") {
      summaryShaps <- summaryShaps[summaryShaps$shapratio > cutoff, ]
    }
    else if (method == "lowerCI") {
      summaryShaps <- summaryShaps[summaryShaps$lowerCI > cutoff, ]
    }
    else stop("method must be one of 'mean', 'shapratio', or 'ci'")
  }

  # STEP 5: PLOT
  # ============================================================
  summaryShaps$feature <- factor(summaryShaps$feature,
                                 levels = summaryShaps$feature[order(summaryShaps["normalized_mean"])])

  Plot <- ggplot2::ggplot(data = summaryShaps,
                 aes(x = "feature",
                     y = "normalized_mean")) +
    geom_col(fill = "#07B86B", alpha = 0.8) +
    geom_errorbar(aes(ymin = "lowerCI",
                      ymax = "upperCI"),
                  width = 0.2, color = "#7A004BF0",
                  alpha = 0.75, size = 0.7) +
    coord_flip() +  # Rotating the graph to have mean values on X-axis
    ggtitle("") +
    xlab("Features\n") +
    ylab("\nMean absolute SHAP contributions with 95% CI") +
    theme_classic() +
    # Reduce top plot margin
    theme(plot.margin = margin(t = -0.5,
                               r = .25,
                               b = .25,
                               l = .25,
                               unit = "cm")) +
    # Set lower limit of expansion to 0
    scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

  # To plot or not to plot! That is the question...
  # ============================================================
  if (plot) print(Plot)

  obj <- list(ids = ids,
              plot = Plot,
              summaryShaps = summaryShaps,
              feature_importance = feature_importance,
              weights = w,
              results = results,
              shap_contributions_by_ids = results)

  class(obj) <- c("shapley", "list")

  return(obj)
}


# ids <- h2o.get_ids(aml)
# a <- importance(models = ids,
#           newdata = prostate
#           )
# a



