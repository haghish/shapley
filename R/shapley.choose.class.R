# Internal SHAPLEY function.
# ============================================================

shapley.choose.class <- function(shaps, target_class, model_id) {
  if (!is.list(shaps) || is.data.frame(shaps)) return(shaps)
  if (!length(shaps)) stop("No SHAP values returned by model ", model_id)

  if (!is.null(target_class)) {
    if (!is.null(names(shaps)) && target_class %in% names(shaps)) {
      return(shaps[[target_class]])
    }
    if (grepl("^[0-9]+$", as.character(target_class))) {
      k <- as.integer(target_class)
      if (k >= 1 && k <= length(shaps)) return(shaps[[k]])
    }
    stop("target_class is not available in model ", model_id,
         ". Available outputs: ", paste(names(shaps), collapse = ", "))
  }

  if (length(shaps) == 1L) return(shaps[[1]])
  if (length(shaps) == 2L) return(shaps[[2]]) # second class = positive class
  stop("Model ", model_id, " has multiple outputs. Specify 'target_class'.")
}
