#' @title Select the Target Class for SHAP Contributions
#'
#' @description
#' An internal function that selects SHAP contributions for a specified
#' prediction class or output when a model returns multiple sets of SHAP
#' values. The function supports named and position-based selection and
#' automatically selects the positive class for binary classification
#' models when no target class is specified.
#'
#' When \code{target_class} is specified, the function first attempts
#' to match it to the names of the available outputs. If no matching
#' name is found and \code{target_class} represents a positive integer,
#' the corresponding list element is selected by its position.
#' An error is raised if the requested class or output is unavailable.
#'
#' When \code{target_class = NULL}, the function automatically selects
#' the first output if only one is available, or the second output
#' if exactly two are available, treating the second output as the
#' positive class. For models with more than two outputs,
#' \code{target_class} must be specified explicitly.
#'
#' @keywords internal
#' @noRd


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
