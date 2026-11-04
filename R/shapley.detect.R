# Internal SHAPLEY function.
# ============================================================

shapley.detect <- function(model, shap_backend, background_data,
                                   target_class = NULL) {
  # shapr is an explicit alternative. It is not selected automatically,
  # because conditional SHAP and marginal SHAP are different quantities.
  if (shap_backend != "auto") {
    if (shap_backend == "gpboost" && !inherits(model, "gpb.Booster")) {
      stop("The gpboost backend requires 'gpb.Booster' models")
    }
    return(shap_backend)
  }

  # GPBoost includes a native tree-contribution implementation.
  # Its SHAP output explains the boosted-tree component, not random effects.
  # ============================================================
  if (inherits(model, "gpb.Booster")) return("gpboost")
  if (inherits(model, "GPModel")) {
    stop("A standalone GPModel has no native TreeSHAP representation. ",
         "To explain its predictions, use a model-agnostic method with ",
         "an explicit prediction function and reference data.")
  }

  tree_classes <- c("ranger", "randomForest", "gbm", "xgb.Booster", "lgb.Booster")
  if (!inherits(model, tree_classes)) return("kernelshap")

  # In treeshap >= 0.4.0, probability ranger forests use pred.1.
  if (inherits(model, "ranger")) {
    if (identical(model$treetype, "Probability estimation")) {
      version_ok <- requireNamespace("treeshap", quietly = TRUE) &&
        utils::packageVersion("treeshap") >= "0.4.0"
      class_ok <- "1" %in% as.character(model$forest$levels)
      target_ok <- is.null(target_class) || identical(as.character(target_class), "1")
      if (!version_ok || !class_ok || !target_ok) return("kernelshap")
    } else if (!identical(model$treetype, "Regression")) {
      return("kernelshap")
    }
  }
  if (inherits(model, "randomForest") && identical(model$type, "classification")) {
    return("kernelshap")
  }
  if (inherits(model, c("ranger", "randomForest", "gbm")) &&
      any(vapply(background_data, function(z) is.factor(z) || is.character(z), logical(1)))) {
    return("kernelshap")
  }
  if (!requireNamespace("treeshap", quietly = TRUE)) return("kernelshap")
  return("treeshap")
}
