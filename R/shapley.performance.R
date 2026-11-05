#' @title Prepare Model-Performance Scores
#'
#' @description Validates and organizes model-performance scores used
#'   to calculate WMSHAP. Ensures that
#'   each model has a corresponding performance score and that the
#'   scores are correctly matched to the model IDs.
#'
#' @param model_performance Numeric vector containing the performance
#'   scores of the fitted models, obtained from
#'   cross-validation.
#' @param ids Character vector containing the model IDs.
#'
#' @return A named numeric vector containing one performance score
#'   per model, ordered according to \code{ids}.
#'
#' @noRd

shapley.performance <- function(model_performance, ids) {
  if (is.null(model_performance)) {
    if (length(ids) == 1L) return(setNames(1, ids))
    stop("For non-H2O models, supply 'model_performance' from cross-validation ",
         "(e.g., AUCPR), in the same order as 'models' or named by model ID.")
  }
  if (!is.numeric(model_performance)) stop("'model_performance' must be numeric")
  if (!is.null(names(model_performance))) {
    if (!all(ids %in% names(model_performance))) {
      stop("Names of 'model_performance' must include every model ID")
    }
    model_performance <- model_performance[ids]
  }
  if (length(model_performance) != length(ids)) {
    stop("'model_performance' must provide one value per model")
  }
  if (any(!is.finite(model_performance))) stop("Invalid model performance values")
  return(setNames(as.numeric(model_performance), ids))
}
