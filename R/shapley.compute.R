#' @title Compute SHAP Values Using a Selected Backend
#'
#' @description
#' Computes observation-level SHAP contributions
#' for a single fitted R model using a specified SHAP backend.
#'
#' @param model A fitted R model
#'
#' @param newdata A data frame containing the observations for which
#'   SHAP contributions are computed.
#'
#' @param background_data A data frame containing reference observations
#'   used for computing SHAP values. This dataset is used to construct
#'   the unified tree representation for TreeSHAP, define the reference
#'   distribution for Kernel SHAP, and estimate feature dependence and
#'   baseline predictions for shapr. This argument is not used by
#'   native GPBoost SHAP or H2O.
#'
#' @param shap_backend Character. Specifies the backend used to compute
#'   SHAP values.
#'
#' @param pred_fun A user-defined prediction function of the form
#'   \code{function(model, newdata)}. The function should return
#'   numeric predictions for the supplied observations. It is used
#'   by Kernel SHAP and shapr when provided. If \code{NULL},
#'   the default prediction method is used when supported.
#'
#' @param target_class Character, numeric, or \code{NULL}. Specifies
#'   the prediction class or output for which SHAP contributions
#'   are computed. When \code{NULL}, the second output is selected
#'   for binary classification models returning two outputs.
#'   For models with more than two outputs, the target class
#'   must be specified explicitly.
#'
#' @param model_id Character. Unique identifier of the fitted model,
#'   used to identify the model in error and warning messages.
#'
#' @param shapr_approach Character vector. Specifies the method used
#'   by \code{shapr::explain()} to estimate conditional feature
#'   distributions. The default is \code{"empirical"}. This argument
#'   is used only when \code{shap_backend = "shapr"}.
#'
#' @param shapr_phi0 Numeric or \code{NULL}. Baseline prediction used
#'   by the shapr backend. If \code{NULL} (default), the baseline
#'   is calculated as the mean model prediction over
#'   \code{background_data}. Otherwise, a single finite numeric
#'   value must be provided.
#'
#' @param shapr_n_MC_samples Integer. Number of Monte Carlo samples
#'   used by \code{shapr::explain()}. The default is \code{1000L}.
#'   This argument is used only when \code{shap_backend = "shapr"}.
#'
#' @return A numeric matrix containing observation-level SHAP
#'   contributions, with observations in rows and predictors in
#'   columns.
#'
#' @seealso \code{\link{shapley}}, \code{\link{shapley.external}}
#'
#' @keywords internal
#' @noRd
#' @author E. F. Haghish


shapley.compute <- function(model, newdata, background_data,
                                    shap_backend, pred_fun, target_class,
                                    model_id, shapr_approach = "empirical",
                                    shapr_phi0 = NULL,
                                    shapr_n_MC_samples = 1000L) {
  x <- names(newdata)

  # Compute GPBoost native SHAP for the boosted-tree component.
  # ============================================================
  if (shap_backend == "gpboost") {
    values <- shapley.gpboost(model, newdata, model_id = model_id)
  }

  # Compute TreeSHAP values, if requested.
  # ============================================================
  else if (shap_backend == "treeshap") {
    if (!requireNamespace("treeshap", quietly = TRUE)) {
      stop("Package 'treeshap' must be installed")
    }

    # the ranger package can be problematic
    if (inherits(model, "ranger") && !identical(model$treetype, "Regression")) {
      supported_probability <- identical(model$treetype, "Probability estimation") &&
        utils::packageVersion("treeshap") >= "0.4.0" &&
        "1" %in% as.character(model$forest$levels) &&
        (is.null(target_class) || identical(as.character(target_class), "1"))
      if (!supported_probability) {
        stop("This ranger model is not supported by TreeSHAP. Use 'kernelshap' or 'shapr'.")
      }
    }
    if (inherits(model, "randomForest") && identical(model$type, "classification")) {
      stop("TreeSHAP is not enabled for randomForest classification. Use 'kernelshap' or 'shapr'.")
    }
    if (inherits(model, c("ranger", "randomForest", "gbm")) &&
        any(vapply(background_data, function(z) is.factor(z) || is.character(z), logical(1)))) {
      stop("Categorical predictors are not supported by this TreeSHAP converter.")
    }
    unified <- treeshap::unify(model, data = background_data)
    out <- treeshap::treeshap(unified, newdata, verbose = FALSE)
    if (inherits(out, "treeshap_multioutput")) {
      out <- shapley.choose.class(out, target_class, model_id)
    }
    values <- out$shaps
  }

  # Compute Kernel SHAP values, if requested.
  # ============================================================
  else if (shap_backend == "kernelshap") {
    if (!requireNamespace("kernelshap", quietly = TRUE)) {
      stop("Package 'kernelshap' must be installed")
    }
    args <- list(object = model, X = newdata,
                 bg_X = background_data, verbose = FALSE,
                 parallel = FALSE)
    if (inherits(model, "ranger") && identical(model$treetype, "Classification") &&
        is.null(pred_fun)) {
      stop("For ranger classification, fit probability = TRUE or specify 'pred_fun'.")
    }
    if (inherits(model, "glm") && is.null(pred_fun)) {
      pred_fun <- function(model, X, ...) {
        stats::predict(model, newdata = X, type = "response", ...)
      }
    }
    if (inherits(model, "randomForest") && identical(model$type, "classification") &&
        is.null(pred_fun)) {
      pred_fun <- function(model, X, ...) {
        stats::predict(model, newdata = X, type = "prob", ...)
      }
    }
    # Without a user-defined prediction function, GPBoost is explained
    # on the raw tree-only scale, consistently with its native SHAP values.
    if (inherits(model, "gpb.Booster") && is.null(pred_fun)) {
      pred_fun <- function(model, X, ...) {
        if (any(!vapply(X, is.numeric, logical(1)))) {
          stop("GPBoost requires numeric predictors")
        }
        stats::predict(model, data = as.matrix(X),
                       pred_latent = TRUE, ignore_gp_model = TRUE)
      }
    }
    if (!is.null(pred_fun)) args$pred_fun <- pred_fun
    out <- do.call(kernelshap::kernelshap, args)
    if (!is.null(out$converged) && any(!out$converged)) {
      warning("Kernel SHAP did not converge for all observations in model ", model_id)
    }
    values <- shapley.choose.class(out$S, target_class, model_id)
  }

  # Compute dependence-aware SHAP values using shapr.
  # ============================================================
  else if (shap_backend == "shapr") {
    if (!requireNamespace("shapr", quietly = TRUE)) {
      stop("The 'shapr' backend requires the optional package 'shapr'. ",
           "Install it with install.packages('shapr').")
    }
    if (utils::packageVersion("shapr") < "1.1.0") {
      stop("The 'shapr' backend requires shapr >= 1.1.0.")
    }

    # shapr requires a single numeric prediction for each observation.
    # Its prediction function is called 'predict_model', rather than 'pred_fun'.
    predict_model <- function(model, newdata) {
      shapley.predict(model, as.data.frame(newdata),
                              pred_fun = pred_fun,
                              target_class = target_class,
                              model_id = model_id)
    }

    # phi0 is the expected prediction over the reference observations.
    if (is.null(shapr_phi0)) {
      phi0 <- mean(predict_model(model, background_data))
    } else {
      phi0 <- shapr_phi0
    }
    if (!is.numeric(phi0) || length(phi0) != 1L || !is.finite(phi0)) {
      stop("'shapr_phi0' must be one finite numeric value for model ", model_id)
    }

    out <- shapr::explain(
      model = model,
      x_explain = newdata,
      x_train = background_data,
      approach = shapr_approach,
      phi0 = phi0,
      predict_model = predict_model,
      n_MC_samples = as.integer(shapr_n_MC_samples),
      scope = "local",
      extra_computation_args = list(vS_batching_method = "forloop"),
      verbose = NULL
    )

    # 'none' contains phi0, and 'explain_id' identifies observations.
    # Neither column is a feature contribution.
    if (is.null(out$shapley_values_est)) {
      stop("shapr did not return shapley_values_est for model ", model_id)
    }
    values <- as.data.frame(out$shapley_values_est)
    if ("explain_id" %in% names(values)) {
      index <- as.integer(values$explain_id)
      if (!setequal(index, seq_len(nrow(newdata))) || anyDuplicated(index)) {
        stop("shapr returned unexpected observation indices for model ", model_id)
      }
      values <- values[order(index), , drop = FALSE]
    }
    values <- values[, !names(values) %in% c("none", "explain_id"), drop = FALSE]
  }
  else stop("Invalid SHAP backend: ", shap_backend)

  # Standardize the output for WMSHAP computation.
  # ============================================================
  values <- as.matrix(values)
  if (!is.numeric(values) || nrow(values) != nrow(newdata)) {
    stop("Invalid SHAP contribution matrix for model ", model_id)
  }
  if (is.null(colnames(values)) && ncol(values) == length(x)) {
    colnames(values) <- x
  }
  if (!setequal(colnames(values), x)) {
    stop("The SHAP predictors do not match 'newdata' for model ", model_id)
  }
  values <- values[, x, drop = FALSE]
  if (any(!is.finite(values))) stop("Non-finite SHAP values in model ", model_id)
  return(values)
}
