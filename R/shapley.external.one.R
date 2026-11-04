# Internal SHAPLEY function.
# Computes local SHAP contributions for one fitted R model.
# ============================================================

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
