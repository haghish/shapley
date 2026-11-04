# Internal SHAPLEY function.
# ============================================================

shapley.models <- function(models) {
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
