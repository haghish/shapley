# check sample_size argument

#' @title Weighted Mean SHAP (WMSHAP) and Confidence Interval for Multiple Models
#'        (tuning grid, stacked ensemble, etc.)
#'
#' @description
#' Computes Weighted Mean SHAP ratios (WMSHAP) and confidence intervals to assess feature
#' importance across a collection of models (e.g., an H2O grid/AutoML leaderboard or
#' base-learners of an ensemble). Instead of reporting SHAP contributions for a single model,
#' this function summarizes feature importance across multiple models and weights each model
#' by a chosen performance metric.
#' Currently, only models trained by the \code{h2o} machine learning platform,
#' \code{autoEnsemble}, and the \code{HMDA} R packages are supported.
#'
#' @details
#'    The function works as follows:
#'    \enumerate{
#'      \item For each model, SHAP contributions are computed on \code{newdata}.
#'      \item For each model, feature-level absolute SHAP contributions are aggregated and
#'            converted to a \emph{ratio} (share of total absolute SHAP across features).
#'      \item Models are weighted by a performance metric (e.g., \code{"r2"} for regression or
#'            \code{"auc"} / \code{"aucpr"} for classification).
#'      \item The weighted mean SHAP ratio (WMSHAP) is computed for each feature, along with an
#'            confidence interval across models.
#'    }
#'
#' @param models An H2O AutoML object, H2O grid object, \code{autoEnsemble} object,
#'               or a character vector of H2O model IDs.
#' @param newdata An \code{H2OFrame} (i.e., a \code{data.frame}) already uploaded to the
#'                \code{h2o} server. SHAP contributions are computed on this data.
#' @param performance_metric Character. Performance metric used to weight models.
#'                           Options are \code{"r2"} (regression), \code{"aucpr"}, \code{"auc"}, and \code{"f2"}
#'                           (classification metrics).
#' @param standardize_performance_metric Logical. If \code{TRUE}, rescales model weights so
#'                                       the weights sum to the number of included models.
#'                                       The default is \code{FALSE}.
#' @param performance_type Character. Specify which performance metric performance estimate to use:
#'                         \code{"train"} for training data, \code{"valid"}
#'                        for validation, or \code{"xval"} for cross-validation (default).
#' @param minimum_performance Numeric. Specify the minimum performance metric
#'                            for a model to be included in calculating WMSHAP.
#'                            Models below this threshold receive
#'                            zero weight and are excluded. The default is \code{0}.
#'                            Specifying a minimum performance can be used to compute
#'                            WMSHAP for a set of competitive models.
#' @param method Character. Specify the method for selecting important features
#'               based on their WMSHAP. The default is
#'               \code{"mean"}, which selects features whose WMSHAP
#'               exceeds the \code{cutoff}. The alternative is
#'               \code{"lowerCI"}, which selects features whose lower bound of confidence
#'               interval exceeds the \code{cutoff}.
#' @param cutoff Numeric. Cutoff applied by \code{method} for selecting important features.
#' @param top_n_features Integer or \code{NULL}. If not \code{NULL}, restricts SHAP computation to the
#'                       top N features per model (reduces runtime). This also selects the top N features by WMSHAP
#'                       in the returned \code{selectedFeatures}.
#' @param n_models Integer. Minimum number of models that must meet the performance threshold
#'                 for WMSHAP and CI computation. Use \code{1} to compute summary SHAP for a single model.
#'                 The default is 10.
#' @param sample_size Integer. Number of rows in \code{newdata} used for SHAP assessment.
#'                    Defaults to all rows. Reducing this can speed up development runs.
#' @param plot Logical. If \code{TRUE}, plots the WMSHAP summary (via \code{shapley.plot()}).
#' @importFrom utils setTxtProgressBar txtProgressBar globalVariables
#' @importFrom stats weighted.mean
#' @importFrom h2o h2o.getModel h2o.auc h2o.aucpr h2o.r2
#'             h2o.F2 h2o.shap_summary_plot
#' @return An object of class \code{"shapley"} (a named list) containing:
#' \describe{
#'   \item{ids}{Character vector of model IDs originally supplied or extracted.}
#'   \item{included_models}{Character vector of model IDs included after filtering by performance.}
#'   \item{ignored_models}{Data frame of excluded models and their performance.}
#'   \item{weights}{Numeric vector of model weights (performance metrics) for included models.}
#'   \item{results}{Data frame of row-level SHAP contributions merged across models.}
#'   \item{summaryShaps}{Data frame of feature-level WMSHAP means and confidence intervals.}
#'   \item{selectedFeatures}{Character vector of selected important features.}
#'   \item{feature_importance}{List of per-feature absolute contribution summaries by model.}
#'   \item{contributionPlot}{A ggplot-like object returned by \code{h2o.shap_summary_plot()}
#'         used for the WMSHAP (“wmshap”) style plot.}
#'   \item{plot}{A ggplot object (bar plot) if \code{plot = TRUE}, otherwise \code{NULL}.}
#' }
#'
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
                    sample_size = NULL
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

  # if sample size is NULL, use all the data. otherwise, make a random subset
  # from the data and use it for the computation
  # ============================================================
  if (is.null(sample_size)) {
    sample_size <- nrow(newdata)
  }
  else {
    newdata <- newdata[sample.int(n = nrow(newdata), size = sample_size, replace = FALSE), ]
    sample_size <- NULL     #Reset it because it is not needed anymore
  }

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
  if (inherits(models,"H2OAutoML") | inherits(models,"H2OGrid")) {
    ids <- h2o.get_ids(models)
  }
  else if (inherits(models,"autoEnsemble")) {
    ids <- models[["top_rank_id"]]
  }
  else if (inherits(models,"character")) {
    ids <- models
  }
  else {
    stop("`models` must be an H2OAutoML, H2OGrid, autoEnsemble, or a character vector of model IDs.", call. = FALSE)
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
          sample_size = sample_size
        )
      }
      else {
        m <- h2o.shap_summary_plot(
          model = model,
          newdata = newdata,
          columns = model@allparameters$x, #get SHAP for all predictors
          sample_size = sample_size
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
    X <- ratioDF[ratioDF$feature == j, grep("^ratio", names(ratioDF)), FALSE]
    X  <- as.numeric(X[1, ])   #get a numeric vector of contributions

    # make sure x is numeric and performance (w) is non-negative
    ok <- is.finite(X) & is.finite(w) & (w >= 0)
    X <- X[ok]
    w_ok <- w[ok]

    weighted_mean <- weighted.mean(X, w_ok, na.rm = TRUE)

    # Uses weights as given (NO scaling/normalization)
    if (sum(w_ok) <= 1) {
      weighted_sd <- NA_real_
      ci <- NA_real_
    } else {
      weighted_var  <- sum(w_ok * (X - weighted_mean)^2, na.rm = TRUE)  /  (sum(w_ok, na.rm = TRUE) - 1)
      weighted_sd   <- sqrt(weighted_var)
    }

    # update the summaryShaps data frame
    summaryShaps[summaryShaps$feature == j, "mean"] <- weighted_mean #mean(X)
    summaryShaps[summaryShaps$feature == j, "sd"] <- weighted_sd
    summaryShaps[summaryShaps$feature == j, "ci"] <- 1.96 * weighted_sd / sqrt(length(X))
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
  BASE$labels$title <- "WMSHAP Summary Plot\n"

  # STEP 3: NORMALIZE the SHAP contributions and their CI
  # ============================================================
  #???

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

