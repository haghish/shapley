#' @title Compute SHAP Values for a Single H2O Model
#'
#' @description
#' An internal function that computes SHAP contributions and evaluates the
#' predictive performance of a single H2O model. It retrieves the fitted
#' model, calculates its performance using the specified metric, and obtains
#' observation-level SHAP values using H2O's native SHAP implementation.
#'
#' For models exceeding the minimum performance threshold, SHAP values
#' are computed using \code{h2o::h2o.predict_contributions()} with
#' \code{output_format = "compact"}. The resulting SHAP contributions are
#' organized into a matrix with observations in rows and predictors in
#' columns. The bias term is excluded, and predictors absent from a
#' particular model are assigned zero contributions. This ensures that
#' SHAP values from different models can be combined using a common set
#' of predictors.
#'
#' @param model_id Character. Unique identifier of a fitted H2O model,
#'   which is retrieved using \code{h2o::h2o.getModel()}.
#'
#' @param shared_data A named list containing information shared across
#'   models. The default is \code{NULL}, in which case the function
#'   retrieves \code{.shapley_shared_data} from the global environment.
#'   The list must contain \code{frame_id}, identifying the H2OFrame used
#'   for SHAP computation; \code{predictors}, a character vector of
#'   predictor names shared across the analysis.
#'
#' @return A list containing the following elements:
#'   \describe{
#'     \item{id}{Character. Unique identifier of the evaluated H2O model.}
#'     \item{performance}{Numeric. Predictive performance of the model
#'       according to the specified performance metric.}
#'     \item{values}{Numeric matrix containing observation-level SHAP
#'       contributions, with observations in rows and predictors in
#'       columns. Returns \code{NULL} if the model performance is less
#'       than or equal to \code{minimum_performance}.}
#'   }
#'
#' @note
#' The function requires a single finite model performance estimate.
#' If the performance metric is \code{"f2"}, the current implementation
#' uses \code{h2o::h2o.F2()}.
#'
#'
#' @noRd


shapley.h2o.one <- function(model_id, shared_data = NULL) {
  if (is.null(shared_data)) {
    shared_data <- get(".shapley_shared_data", envir = .GlobalEnv)
  }

  model <- h2o::h2o.getModel(model_id)
  metric <- shared_data$performance_metric
  train <- identical(shared_data$performance_type, "train")
  valid <- identical(shared_data$performance_type, "valid")
  xval <- identical(shared_data$performance_type, "xval")

  if (metric == "r2") {
    performance <- h2o::h2o.r2(model, train = train, valid = valid, xval = xval)
  } else if (metric == "aucpr") {
    performance <- h2o::h2o.aucpr(model, train = train, valid = valid, xval = xval)
  } else if (metric == "auc") {
    performance <- h2o::h2o.auc(model, train = train, valid = valid, xval = xval)
  } else if (metric == "f2") {
    metrics <- h2o::h2o.performance(model, train = train, valid = valid, xval = xval)
    performance <- h2o::h2o.F2(metrics, thresholds = "max")[[1L]]
  } else {
    stop("Unsupported performance metric")
  }
  performance <- as.numeric(performance)
  if (length(performance) != 1L || !is.finite(performance)) {
    stop("A single finite performance estimate is required for H2O model: ", model_id)
  }
  if (performance <= shared_data$minimum_performance) {
    return(list(id = model_id, performance = performance, values = NULL))
  }

  # The native contribution function returns one row per observation.
  # Unlike h2o.shap_summary_plot(), it does not construct a plot.
  # ============================================================
  test_hex <- h2o::h2o.getFrame(shared_data$frame_id)
  predictors <- shared_data$predictors
  shap <- h2o::h2o.predict_contributions(
    model, newdata = test_hex[, predictors], output_format = "compact"
  )
  shap <- as.data.frame(shap)

  # Exclude BiasTerm, and retain a common predictor matrix across models.
  # A predictor absent from the fitted model receives zero contribution.
  # ============================================================
  values <- matrix(0, nrow = nrow(shap), ncol = length(predictors),
                   dimnames = list(NULL, predictors))
  available <- intersect(predictors, names(shap))
  if (length(available)) {
    values[, available] <- as.matrix(shap[, available, drop = FALSE])
  }
  missing <- setdiff(as.character(model@allparameters$x), names(shap))
  if (length(missing)) {
    stop("H2O did not return compact SHAP contributions for predictors: ",
         paste(missing, collapse = ", "), ". Model: ", model_id)
  }
  if (any(!is.finite(values))) stop("Non-finite H2O SHAP values in model: ", model_id)

  return(list(id = model_id, performance = performance, values = values))
}
