#' @title Compute SHAP Values for a Single Non-H2O Model
#'
#' @description
#' An internal function that computes observation-level SHAP contributions
#' for a single fitted R model using the specified SHAP backend.
#'
#' @details
#' The function retrieves the fitted model and its assigned SHAP backend
#' from \code{task}, then calls \code{shapley.compute()} to calculate SHAP
#' contributions for the observations in \code{newdata}.
#'
#' @param task A list containing information about the fitted model
#'   and the SHAP backend. The list includes \code{id}, \code{model},
#'   \code{backend}, and \code{phi0}. For GPBoost models evaluated in parallel,
#'   \code{model_file} may specify the path to a saved model, which is
#'   loaded instead of using \code{model}.
#'
#' @param shared_data A named list containing information shared across
#'   models. The default is \code{NULL}, in which case the function
#'   retrieves \code{.shapley_shared_data} from the global environment.
#'
#' @return A list.
#'
#' @note
#' The function computes SHAP contributions for a single model but does
#' not calculate performance weights or aggregate SHAP values across
#' models.
#'
#' @noRd


shapley.external.one <- function(task, shared_data = NULL) {
  if (is.null(shared_data)) {
    shared_data <- get(".shapley_shared_data", envir = .GlobalEnv)
  }
  model <- task$model
  # GPBoost R6 models contain native handles. Workers load a model saved
  # by the main R session rather than receiving an R6 object over a socket.
  if (!is.null(task$model_file)) {
    if (!requireNamespace("gpboost", quietly = TRUE)) {
      stop("The optional package 'gpboost' is required to load GPBoost models. ",
           "Install it with install.packages('gpboost').")
    }
    model <- gpboost::gpb.load(filename = task$model_file)
  }

  values <- shapley.compute(
    model = model,
    newdata = shared_data$newdata,
    background_data = shared_data$background_data,
    shap_backend = task$backend,
    pred_fun = shared_data$pred_fun,
    target_class = shared_data$target_class,
    model_id = task$id,
    shapr_approach = shared_data$shapr_approach,
    shapr_phi0 = task$phi0,
    shapr_n_MC_samples = shared_data$shapr_n_MC_samples
  )
  return(list(id = task$id, values = values))
}
