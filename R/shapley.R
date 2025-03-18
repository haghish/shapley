#' @title Weighted Mean SHAP Ratio and Confidence Interval for a ML Grid
#'        of Fine-Tuned Models or Base-Learners of a Stacked Ensemble Model
#'
#' @description
#' Calculates weighted mean SHAP ratios and confidence intervals to assess feature importance
#' across a collection of models (e.g., a grid of fine-tuned models or base-learners
#' in a stacked ensemble). Rather than reporting relative SHAP contributions for
#' only a single model, this function accounts for variability in feature importance
#' across multiple models. Each model's performance metric is used as a weight.
#' The function also provides a plot of weighted SHAP values with confidence intervals.
#' Currently, only models trained by the \code{h2o} machine learning platform,
#' \code{autoEnsemble}, and the \code{HMDA} R packages are supported.
#'
#' @details
#'    The function works as follows:
#'    \enumerate{
#'      \item SHAP contributions are computed at the individual level (row) for each model for the given "newdata".
#'      \item Each model's feature-level SHAP ratios (i.e., share of total SHAP) are computed.
#'      \item The performance metrics of the models are used as weights.
#'      \item Using the weights vector and shap ratio of features for each model,
#'            the weighted mean SHAP ratios and their confidence intervals are computed.
#'    }
#'
#' @param models h2o search grid, autoML grid, or a character vector of H2O model IDs.
#' @param newdata An \code{h2o} frame (or \code{data.frame}) already uploaded to the
#'                \code{h2o} server. This data will be used for computing SHAP
#'                contributions for each model, alongside model's performance
#'                weights.
#' @param performance_metric Character specifying which performance metric to use
#'                           as weights. The default is \code{"r2"}, which can
#'                           be used for both regression and classification.
#'                           For binary classification, other options include:
#'                           \code{"aucpr"} (area under the precision-recall curve),
#'                           \code{"auc"} (area under the ROC curve),
#'                           and \code{"f2"} (F2 score).
#' @param standardize_performance_metric Logical, indicating whether to standardize
#'                                       the performance metric used as weights so
#'                                       their sum equals the number of models. The
#'                                       default is \code{FALSE}.
#' @param performance_type Character. Specify which performance metric should be
#'                        reported: \code{"train"} for training data, \code{"valid"}
#'                        for validation, or \code{"xval"} for cross-validation (default).
#' @param minimum_performance Numeric. Specify the minimum performance metric
#'                            for a model to be included in calculating weighted
#'                            mean SHAP ratio Models below this threshold receive
#'                            zero weight. The default is \code{0}.
#' @param method Character. Specify the method for selecting important features
#'               based on their weighted mean SHAP ratios. The default is
#'               \code{"mean"}, which selects features whose weighted mean shap ratio (WMSHAP)
#'               exceeds the \code{cutoff}. The alternative is
#'               \code{"lowerCI"}, which selects features whose lower bound of confidence
#'               interval exceeds the \code{cutoff}.
#' @param cutoff numeric, specifying the cutoff for the method used for selecting
#'               the top features.
#' @param top_n_features integer. if specified, the top n features with the
#'                       highest weighted SHAP values will be selected, overrullung
#'                       the 'cutoff' and 'method' arguments. specifying top_n_feature
#'                       is also a way to reduce computation time, if many features
#'                       are present in the data set. The default is NULL, which means
#'                       the shap values will be computed for all features.
#' @param n_models minimum number of models that should meet the 'minimum_performance'
#'                 criterion in order to compute WMSHAP and CI. If the intention
#'                 is to compute global summary SHAP values (at feature level) for
#'                 a single model, set n_models to 1. The default is 10.
#' @param sample_size integer. number of rows in the \code{newdata} that should
#'                    be used for SHAP assessment. By default, all rows are used,
#'                    which is the recommended procedure for scientific analyses.
#'                    However, SHAP analysis is time consuming and in the process
#'                    of code development, lower values can be used for quicker
#'                    shapley analyses.
#' @param plot logical. if TRUE, the weighted mean and confidence intervals of
#'             the SHAP values are plotted. The default is TRUE.
#'
# @param normalize_shap_per_model Logical. If TRUE, the SHAP contribution ratio for each
#                        model is normalized to range between 0 to 1. The default is
#                        FALSE, which encourages reporting Weighted Mean SHAP Ratios (not ranks)
#                        comparison between models without scaling each model.
#                        This option is only provided for rare use cases where
#                        WMSHAP values are prefered to be scaled to 0 to 1, but
#                        should otherwise be avoided.
# @param normalize_to character. The default value is "upperCI", which sets the feature with
#                     the maximum SHAP value to one, allowing the higher CI to
#                     go beyond one. Setting this value is mainly for aesthetic
#                     reason to adjust the Plot, but also, it can influence the
#                     feature selection process, depending on the method in use,
#                     because it changes how the SHAP values should be normalized.
#                     the alternative is 'feature', specifying that
#                     in the normalization of the SHAP values, the maximum confidence
#                     interval of the weighted SHAP values should be equal to
#                     "1", in order to limit the plot values to maximum of one.
#'
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
#' @importFrom h2o h2o.stackedEnsemble h2o.getModel h2o.auc h2o.aucpr h2o.r2
#'             h2o.F2 h2o.mean_per_class_error h2o.giniCoef h2o.accuracy
#'             h2o.shap_summary_plot
# @importFrom h2otools h2o.get_ids
#' @importFrom curl curl
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
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
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)
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
#'                  # this setting ensures the models are comparable for building a meta learner
#'                  seed = 2023, nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' ### call 'shapley' function to compute the weighted mean and weighted confidence intervals
#' ### of SHAP values across all trained models.
#' ### Note that the 'newdata' should be the testing dataset!
#' result <- shapley(models = aml, newdata = prostate, performance_metric = "aucpr", plot = TRUE)
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
#' result2 <- shapley(models = grid, newdata = prostate, performance_metric = "aucpr", plot = TRUE)
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
#'
#' }
#' @export
#' @author E. F. Haghish

shapley <- function(models,
                    newdata,
                    #nboot = NULL,
                    plot = TRUE,
                    performance_metric = "r2",
                    standardize_performance_metric = FALSE,
                    performance_type = "xval",
                    minimum_performance = 0,
                    method = "mean",
                    cutoff = 0.01,
                    top_n_features = NULL,
                    n_models = 10,
                    sample_size = nrow(newdata)
                    #normalize_shap_per_model = FALSE
                    #normalize_to = "upperCI"
) {

  # Variables definitions
  # ============================================================
  BASE <- NULL                                           #contribution SHAP plot
  w <- NULL                                              #performance metric (weight)
  results <- NULL                                        #data frame of SHAP values
  selectedFeatures <- NULL                               #list of selected features
  Plot <- NULL                                           #GGPLOT2 object
  feature_importance <- list()                           #list of feature importance
  z <- 0                                                 #counter for the progress bar

  # models with low minimum_performance are stored in 'ignored_models' data.frame
  ignored_models  <- data.frame(id = character(), performance = numeric())
  included_models <- NULL

  # Where should the performance type be retrieved from?
  train <- FALSE
  valid <- FALSE
  xval  <- FALSE

  #if (normalize_shap_per_model) message("using 'normalize_shap_per_model' is discouraged! This is for rare use only")

  # Syntax check
  # ============================================================
  if (performance_metric != "r2" &
      performance_metric != "aucpr" &
      performance_metric != "auc" &
      performance_metric != "f2") stop("performance metric must be 'r2', 'aucpr', 'auc', or 'f2'")

  if (performance_type == "train")      train <- TRUE
  else if (performance_type == "valid") valid <- TRUE
  else if (performance_type == "xval")  xval  <- TRUE
  else stop("performance_type must be 'train', 'valid', or 'xval'")

  # STEP 0: prepare the models, by either confirming that the models are 'h2o' or 'autoEnsemble'
  #        or by extracting the model IDs from these objects
  # ============================================================
  if (inherits(models,"H2OAutoML") | inherits(models,"H2OAutoML")
      | inherits(models,"H2OGrid")) {
    ids <- h2o.get_ids(models)
  }
  else if (inherits(models,"autoEnsemble")) {
    ids <- models[["top_rank_id"]]
  }
  else if (inherits(models,"character")) {
    ids <- models
  }

  # Initiate the progress bar after identifying the ids
  # ------------------------------------------------------------
  pb <- txtProgressBar(z, length(ids), style = 3)

  # STEP 1: Compute SHAP values and performance metric for each model
  # ============================================================
  for (i in ids) {
    z <- z + 1
    performance <- NULL
    model <- h2o.getModel(i)

    # Compute performance metrics
    # ----------------------------------------------------------
    # for regression and classification
    if (performance_metric == "r2") performance <- h2o.r2(model, train = train, valid = valid, xval = xval)

    # for classification
    else if (performance_metric == "aucpr") performance <- h2o.aucpr(model, train = train, valid = valid, xval = xval)
    else if (performance_metric == "auc") performance <- h2o.auc(model, train = train, valid = valid, xval = xval)
    else if (performance_metric == "f2") performance <- h2o.F2(model)

    # If the performance of the model is below the minimum_performance,
    # ignore processing the model to save runtime
    # ----------------------------------------------------------
    if (performance <= minimum_performance) {
      ignored_models <- rbind(ignored_models, c(i, performance), make.row.names=F)
    }

    # otherwise, continue processing the model
    # ----------------------------------------------------------
    else {
      included_models <- c(included_models, i)
      w <- c(w, performance)

      # if top_n_features is not specified, evaluate SHAP for ALL FEATURES,
      # otherwise, evaluate SHAP for the top_n_features. Evaluating SHAP for
      # all features can be very time consuming. Otherwise, there is no other
      # reason to limit the number of features.
      if (!is.null(top_n_features)) {
        m <- h2o.shap_summary_plot(
          model = model,
          newdata = newdata,
          top_n_features = top_n_features,
          sample_size = nrow(newdata)
        )
      }
      else {
        m <- h2o.shap_summary_plot(
          model = model,
          newdata = newdata,
          columns = model@allparameters$x, #get SHAP for all predictors
          sample_size = nrow(newdata)
        )
      }

      # create the summary dataset
      # ----------------------------------------------------------
      if (length(included_models) == 1) {
        # reserve the first dataset for SHAP cntribution
        # reserve the first model's data for rowmean shap computation
        BASE <- m
        data <- m$data
        data <- data[order(data$Row.names), ]
        results <- data[, c("id.x", "Row.names", "feature", "contribution")]
        names(results)[1] <- "index"
      }
      else if (length(included_models) > 1) {
        holder <- m$data[, c("Row.names", "feature", "contribution")]
        colnames(holder) <- c("Row.names", "feature", paste0("contribution", z))
        holder <- holder[order(holder$Row.names), ]
        #results <- cbind(results, holder[, 2, drop = FALSE])

        # NOTE: instead of cbind, use merge because the number of "important" features
        # are not identical according to different models. therefore, merge the
        # datasets and if a new feature is added, then the value of this this
        # feature for previous models should be zero

        results <- merge(results, holder, by="Row.names", all = T)

        findex <- is.na(results[,"feature.x"])
        results[findex,"feature.x"] <- results[findex,"feature.y"]
        results[,"feature.y"] <- NULL
        names(results)[names(results) == "feature.x"] <- "feature"
      }
    }

    setTxtProgressBar(pb, z)
  }

  # number of included_models must be higher than 1

  if (length(included_models) < n_models) stop("number of models that have met the minimum_performance criteria is too low")

  # Check that the sum of weights are larger than 1 to avoid negative variance computation
  # ============================================================
  if (sum(w) <= 1 & !standardize_performance_metric) stop("sum of model(s) performance weight(s) are lower than 1. enable 'standardize_performance_metric' by setting it to TRUE")
  else if (standardize_performance_metric) w <- w * length(w) / sum(w)

  # notify the user about ignored models
  # ============================================================
  if (nrow(ignored_models) > 0) {
    colnames(ignored_models) <- c("id", "performance")
    warning(paste(nrow(ignored_models),
                  "models did not meet the minimum_performance criteria and were excluded.
                  see 'ignored_models' in the returned shapley object"))
  }

  # ???
  # NOTE: if a feature is not important for a model, then the shap value is NA.
  # replace them with zero
  # ============================================================
  results[is.na(results)] <- 0

  # STEP 2: Calculate the summary shap values for each feature and store the mean
  #         SHAP values in a list, for significance testing
  # ============================================================
  ratioDF <- NULL
  UNQ     <- unique(results$feature)
  summaryShaps <- data.frame(
    feature = UNQ,
    mean = NA,
    sd = NA,
    ci = NA,
    lowerCI = NA,
    upperCI = NA)

  # CALCULATE THE TOTAL CONTRIBUTION PER MODEL (don't collapse to vector)
  TOTAL <- colSums(abs(results[, grep("^contribution", names(results)), FALSE]),
                   na.rm = TRUE)

  # Calculate the ratio of contribution of each feature per model
  # -------------------------------------------------------------
  for (j in UNQ) {
    # get all contribution columns for the j feature
    tmp <- results[results$feature == j, grep("^contribution", names(results)), FALSE]
    # compute the ratio of absolute shap values for features of all models
    mat <- matrix(colSums(abs(tmp), na.rm = TRUE) / TOTAL, nrow = 1)
    # create a matrix
    ratioDF <- rbind(ratioDF, mat)
  }

  # Scale the ratio matrix and create a data frame
  # -------------------------------------------------------------
  for (i in 1:ncol(ratioDF)) ratioDF[,i] <- abs(ratioDF[,i])
  ratioDF <- as.data.frame(ratioDF)
  names(ratioDF) <- paste0("ratio", 1:ncol(ratioDF))
  feature <- unique(summaryShaps$feature)
  ratioDF <- cbind(feature, ratioDF)

  # Compute the weighted mean, sd, and ci for each feature
  # -------------------------------------------------------------
  for (j in UNQ) {
    # get all contribution columns for the j feature
    tmp <- ratioDF[ratioDF$feature == j, grep("^ratio", names(ratioDF)), FALSE]
    weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
    weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
    weighted_sd   <- sqrt(weighted_var)

    # update the summaryShaps data frame
    summaryShaps[summaryShaps$feature == j, "mean"] <- weighted_mean #mean(tmp)
    summaryShaps[summaryShaps$feature == j, "sd"] <- weighted_sd
    summaryShaps[summaryShaps$feature == j, "ci"] <- 1.96 * weighted_sd / sqrt(length(tmp))
  }

  # Compute the lower and upper confidence intervals
  # -------------------------------------------------------------
  summaryShaps$lowerCI <- summaryShaps$mean - summaryShaps$ci
  summaryShaps$upperCI <- summaryShaps$mean + summaryShaps$ci

  # compute feature_importance used by shapley.test function
  # -------------------------------------------------------------
  for (j in unique(results$feature)) {
    tmp <- results[results$feature == j, grep("^contribution", names(results)), FALSE]
    tmp <- colSums(abs(tmp))
    feature_importance[[j]] <- tmp
  }

  # Compute row means of SHAP contributions for each subject
  # ============================================================
  cols <- grep("^contribution", names(results))

  # for (r in 1:nrow(data)) {
  #   data[r, "contribution"] <- weighted.mean(results[r, cols], w)
  # }
  #???
  BASE$data <- BASE$data[order(BASE$data$Row.names), ]
  BASE$data$contribution <- data$contribution
  BASE$labels$title <- "SHAP Mean Summary Plot\n"

  # STEP 3: NORMALIZE the SHAP contributions and their CI
  # ============================================================
  # the minimum contribution should not be normalized as zero. instead,
  # it should be the ratio of minimum value to the maximum value.
  # The maximum would be the highest mean + the highest CI

  # if (normalize_to == "upperCI") {
  #   max  <- max(summaryShaps$mean + summaryShaps$ci)
  # }
  # else {
  #   max  <- max(summaryShaps$mean)
  # }

  # #??? I might still give the minimum value to be zero!
  # min  <- 0 # min(summaryShaps$mean)/max
  #
  # summaryShaps$normalized_mean <- normalize(x = summaryShaps$mean,
  #                                           min = min,
  #                                           max = max)
  #
  # summaryShaps$normalized_ci <- normalize(x = summaryShaps$ci,
  #                                         min = min,
  #                                         max = max)
  # compute relative shap values
  # summaryShaps$shapratio <- summaryShaps$mean / sum(summaryShaps$mean)

  # # compute lowerCI
  # summaryShaps$lowerCI <- summaryShaps$normalized_mean - summaryShaps$normalized_ci
  # summaryShaps$upperCI <- summaryShaps$normalized_mean + summaryShaps$normalized_ci

  # STEP 4: Feature selection
  # ============================================================
  selectedFeatures <- summaryShaps[order(summaryShaps$mean, decreasing = TRUE), ]
  if (!is.null(top_n_features)) {
    selectedFeatures <- selectedFeatures[1:top_n_features, ]
  }
  else {
    if (method == "mean") {
      selectedFeatures <- selectedFeatures[selectedFeatures$mean > cutoff, ]
    }
    # else if (method == "shapratio") {
    #   selectedFeatures <- selectedFeatures[selectedFeatures$shapratio > cutoff, ]
    # }
    else if (method == "lowerCI") {
      selectedFeatures <- selectedFeatures[selectedFeatures$lowerCI > cutoff, ]
    }
    else stop("method must be one of 'mean' or 'lowerCI'")
  }


  # STEP 5: Create the shapley object
  # ============================================================
  obj <- list(ids = ids,
              plot = Plot,
              contributionPlot = BASE,
              summaryShaps = summaryShaps,
              selectedFeatures = selectedFeatures$feature,
              feature_importance = feature_importance,
              weights = w,
              results = results,
              shap_contributions_by_ids = results,
              ignored_models = ignored_models,
              included_models= included_models)

  class(obj) <- c("shapley", "list")

  # STEP 6: PLOT
  # ============================================================
  if (plot) {
    obj$plot <- shapley.plot(obj,
                             plot = "bar",
                             method = method,
                             cutoff = cutoff,
                             top_n_features = top_n_features)
    print(obj$plot)
  }

  return(obj)
}

