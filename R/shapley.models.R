
#' @title Prepare Machine Learning Models
#'
#' @description Identifies the type of machine learning models supplied
#'   to shapley and organizes non-H2O models into a named list. The function
#'   also validates model names and generates names when they are missing.
#'
#' @param models A fitted machine learning model, a list of fitted models,
#'   or an H2O Grid, AutoML, autoEnsemble, or vector of H2O model IDs.
#'
#' @return A list containing:
#' \describe{
#'   \item{h2o}{Logical indicating whether the input contains H2O models.}
#'   \item{models}{Named list of fitted non-H2O models, or NULL for H2O inputs.}
#' }
#'
#' @noRd

shapley.models <- function(models) {
  # if models are h2o or just model_ids
  if (inherits(models, c("H2OGrid", "H2OAutoML", "autoEnsemble")) || is.character(models)) {
    return(list(h2o = TRUE, models = NULL))
  }

  if (is.list(models) && identical(class(models), "list")) {
    if (length(models) == 0L) stop("'models' cannot be empty")
    if (is.null(names(models)) || any(is.na(names(models)) | !nzchar(names(models)))) {
      names(models) <- paste0("model", seq_along(models))
    }
    if (anyDuplicated(names(models))) stop("model names must be unique")
    return(list(h2o = FALSE, models = models))
  }

  return(list(h2o = FALSE, models = list(model1 = models)))
}
