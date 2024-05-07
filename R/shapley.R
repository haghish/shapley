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
#' @param models H2O search grid, AutoML grid, or a character vector of H2O model IDs.
#'               the \code{"h2o.get_ids"} function from \code{"h2otools"} can retrieve
#'               the IDs from grids.
#' @param newdata h2o frame (data.frame). the data.frame must be already uploaded
#'                on h2o server (cloud). when specified, this dataset will be used
#'                for evaluating the models. if not specified, model performance
#'                on the training dataset will be reported.
#' @param performance_metric character, specifying the performance metric to be
#'                           used for weighting the SHAP values (mean and 95% CI). The default is
#'                           "r2" (R Squared).
#'                           For binary classification, other options include
#'                           "aucpr" (area under the precision-recall curve),
#'                           "auc" (area under the ROC curve),
#'                           and "f2" (F2 score).
#' @param standardize_performance_metric logical. if TRUE, performance_metric, which is
#'                                       used as weights vector is standardized such
#'                                       that the sum of the weights vector would be equal
#'                                       to the length of the vector. the default value
#'                                       is FALSE.
#' @param performance_type character, specifying where the performance metric should
#'                         be retrieved from. "train" means the performance of the
#'                         training process should be reported, "valid" indicates that
#'                         the performance of the validation process should be reported,
#'                         and "xval" means the cross-validation performance to be
#'                         retrieved.
#' @param minimum_performance the minimum performance metric for a recognizable model.
#'                            any model with performance equal or lower than this
#'                            argument will have weight of zero in computing the
#'                            weighted mean and CI SHAP values. the default value is
#'                            zero.
#' @param method character, specifying the method used for identifying the most
#'               important features according to their weighted SHAP values.
#'               The default selection method is "lowerCI", which includes
#'               features whose lower weighted confidence interval exceeds the
#'               predefined 'cutoff' value (default is 0).
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
#'               the top features. the default is zero, which means that all
#'               features with the "method" criteria above zero will be selected.
#' @param top_n_features integer. if specified, the top n features with the
#'                       highest weighted SHAP values will be selected, overrullung
#'                       the 'cutoff' and 'method' arguments. specifying top_n_feature
#'                       is also a way to reduce computation time, if many features
#'                       are present in the data set. The default is NULL, which means
#'                       the shap values will be computed for all features.
#' @param plot logical. if TRUE, the weighted mean and confidence intervals of
#'             the SHAP values are plotted. The default is TRUE.
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
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
#' @importFrom h2o h2o.stackedEnsemble h2o.getModel h2o.auc h2o.aucpr h2o.r2
#'             h2o.F2 h2o.mean_per_class_error h2o.giniCoef h2o.accuracy
#'             h2o.shap_summary_plot
# @importFrom h2otools h2o.get_ids
#' @importFrom curl curl
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
#' @author E. F. Haghish
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

shapley <- function(models,
                    newdata,
                    n = NULL,
                    plot = TRUE,
                    performance_metric = "r2",
                    standardize_performance_metric = FALSE,
                    performance_type = "xval",
                    minimum_performance = 0,
                    method = c("lowerCI"),
                    cutoff = 0.0,
                    top_n_features = NULL
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
        results <- data[, c("Row.names", "feature", "contribution")]
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

  #data <<- cbind(data, results[, -1])

  # STEP 2: Calculate the summary shap values for each feature and store the mean
  #         shap values in a list, for significance testing
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

  # CALCULATE THE TOTAL CONTRIBUTION PER MODEL
  TOTAL <- colSums(abs(results[, grep("^contribution", names(results))]),
                   na.rm = TRUE)

#globalVariables(c("feature", "mean", "sd", "ci", "normalized_mean", "normalized_ci"))

  # Calculate the ratio of contribution of each feature per model
  # -------------------------------------------------------------
  for (j in UNQ) {
    # get all contribution columns for the j feature
    tmp <- results[results$feature == j, grep("^contribution", names(results))]
    # compute the ratio of absolute shap values for features of all models
    mat <- matrix(colSums(abs(tmp), na.rm = TRUE) / TOTAL, nrow = 1)
    # create a matrix
    ratioDF <- rbind(ratioDF, mat)
  }

  # Scale the ratio matrix and create a data frame
  # -------------------------------------------------------------
  for (i in 1:ncol(ratioDF)) {
    ratioDF[,i] <- normalize(abs(ratioDF[,i]), min = 0)
  }
  ratioDF <- as.data.frame(ratioDF)
  names(ratioDF) <- paste0("ratio", 1:ncol(ratioDF))
  feature <- unique(summaryShaps$feature)              #??? summaryShaps should be empty by now???
  ratioDF <- cbind(feature, ratioDF)

  # Cmpute the weighted mean, sd, and ci for each feature
  # -------------------------------------------------------------
  for (j in UNQ) {
    # get all contribution columns for the j feature
    tmp <- ratioDF[ratioDF$feature == j, grep("^ratio", names(ratioDF))]
    weighted_mean <- weighted.mean(tmp, w, na.rm = TRUE)
    weighted_var  <- sum(w * (tmp - weighted_mean)^2, na.rm = TRUE)  /  (sum(w, na.rm = TRUE) - 1)
XX<<-    weighted_var
print(weighted_var)
    weighted_sd   <- sqrt(weighted_var)
print(weighted_sd)
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
    tmp <- results[results$feature == j, grep("^contribution", names(results))]
    tmp <- colSums(abs(tmp))
    feature_importance[[j]] <- tmp
# weighted_mean <- weighted.mean(tmp, w)
# weighted_var  <- sum(w * (tmp - weighted_mean)^2) / (sum(w) - 1)
# weighted_sd   <- sqrt(weighted_var)
#
# # update the summaryShaps data frame
# summaryShaps[summaryShaps$feature == j, "mean"] <- weighted_mean #mean(tmp)
# summaryShaps[summaryShaps$feature == j, "sd"] <- weighted_sd
# summaryShaps[summaryShaps$feature == j, "ci"] <- 1.96 * weighted_sd / sqrt(length(tmp))
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
  summaryShaps$shapratio <- summaryShaps$mean / sum(summaryShaps$mean)

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
    else if (method == "shapratio") {
      selectedFeatures <- selectedFeatures[selectedFeatures$shapratio > cutoff, ]
    }
    else if (method == "lowerCI") {
      selectedFeatures <- selectedFeatures[selectedFeatures$lowerCI > cutoff, ]
    }
    else stop("method must be one of 'mean', 'shapratio', or 'ci'")
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

