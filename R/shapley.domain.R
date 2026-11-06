# ??? The running speed of this function can be improved

#' @title Compute and plot weighted mean SHAP contributions at group level (factors or domains)
#' @description Aggregates SHAP contributions across user-defined domains (groups of features),
#'              computes weighted means and descriptive between-model
#'              intervals, and returns a plot plus summary tables.
#' @param shapley Object of class \code{"shapley"}, as returned by the 'shapley' function
#' @param domains Named list of character vectors. Each element name is a domain name;
#'                each element value is a character vector of feature names assigned to that domain.
#' @param plot Logical. If \code{TRUE}, a bar plot of domain WMSHAP contributions is created.
#' @param colorcode Character vector for specifying the color names for each domain in the plot.
#' @param print Logical. If TRUE, prints the domain WMSHAP summary table.
#' @param xlab Character. Specify the ggplot 'xlab' label in the plot (default is "Domains")
#' @importFrom stats na.omit aggregate formula
#' @importFrom h2o h2o.shap_summary_plot h2o.getModel
#' @importFrom ggplot2 theme guides guide_legend guide_colourbar
#'             margin theme_classic ylab xlab ggtitle
#'
#' @author E. F. Haghish
#' @export

shapley.domain <- function(shapley, domains, plot = TRUE, print = FALSE,
                           colorcode = NULL, xlab = "Domains") {

  # Basic checks
  # ============================================================
  if (length(shapley[["ids"]]) < 1L) stop("no model ID was found", call. = FALSE)

  if (!is.list(domains) || length(domains) == 0L || is.null(names(domains)) ||
      anyNA(names(domains)) || any(!nzchar(names(domains))) || anyDuplicated(names(domains))) {
    stop("'domains' must be a non-empty named list with unique names", call. = FALSE)
  }

  if (!all(vapply(domains, function(x) is.character(x) && length(x) > 0L && !anyNA(x), logical(1)))) {
    stop("each domain must contain a non-empty character vector of features", call. = FALSE)
  }

  # Check for overlapping domains
  all_features <- unlist(domains, use.names = FALSE)
  if (anyDuplicated(all_features)) {
    stop("each feature must belong to only one domain", call. = FALSE)
  }

  # Variable definitions
  # ============================================================
  # COLORCODE <- c(
  #   "#855C75FF", "#D9AF6BFF", "#AF6458FF", "#736F4CFF",
  #   "#526A83FF", "#625377FF", "#68855CFF", "#9C9C5EFF",
  #   "#A06177FF", "#8C785DFF", "#467378FF", "#7C7C7CFF",
  #   "#B07B52FF", "#5C8D85FF", "#9470A1FF", "#839B5DFF",
  #   "#C17683FF", "#5D86A9FF", "#A08D48FF", "#637D63FF",
  #   "#B48B9BFF", "#9A6B45FF", "#7F90B4FF", "#AD805FFF",
  #   "#7C9B8DFF", "#B17A9EFF", "#9E9C78FF", "#5A788DFF",
  #   "#BA907AFF", "#8A819DFF"
  # )

  COLORCODE <- c(
    "#625377FF", "#D9AF6BFF", "#5C8D85FF", "#AF6458FF",
    "#7F90B4FF", "#736F4CFF", "#B17A9EFF", "#839B5DFF",
    "#526A83FF", "#BA907AFF", "#7C7C7CFF", "#A08D48FF",
    "#9A6B45FF", "#467378FF", "#8A819DFF", "#855C75FF",
    "#637D63FF", "#C17683FF", "#9E9C78FF", "#8C785DFF",
    "#5D86A9FF", "#9470A1FF", "#7C9B8DFF", "#B48B9BFF",
    "#A06177FF", "#B07B52FF", "#9C9C5EFF", "#5A788DFF",
    "#68855CFF", "#AD805FFF"
  )

  if (!is.null(colorcode)) {
    if (!is.character(colorcode) || length(colorcode) == 0L) {
      stop("'colorcode' must be a non-empty character vector", call. = FALSE)
    }
    COLORCODE <- colorcode
  }

  Plot <- NULL

  # Prepare domain data
  # ============================================================
  results <- as.data.frame(shapley$results)
  results$Row.names <- NULL

  if (!all(c("index", "feature") %in% names(results))) {
    stop("'shapley$results' must contain 'index' and 'feature'", call. = FALSE)
  }

  results$feature <- as.character(results$feature)
  contributionNames <- grep("^contribution", names(results), value = TRUE)

  if (length(contributionNames) == 0L) {
    stop("no SHAP contribution columns were found", call. = FALSE)
  }

  if (!all(vapply(results[contributionNames], is.numeric, logical(1)))) {
    stop("SHAP contribution columns must be numeric", call. = FALSE)
  }

  # Assign features to their domains without overwriting the observation index
  # ============================================================
  results$domain <- NA_character_
  missing_features <- setdiff(all_features, unique(results$feature))

  if (length(missing_features) > 0L) {
    stop(paste("Features not found:", paste(missing_features, collapse = ", ")), call. = FALSE)
  }

  for (i in seq_along(domains)) {
    results$domain[results$feature %in% domains[[i]]] <- names(domains)[i]
  }

  # Keep only features assigned to a domain
  # ============================================================
  results <- results[!is.na(results$domain), c("index", "domain", contributionNames), drop = FALSE]

  if (anyNA(results)) stop("domain contributions contain missing values", call. = FALSE)
  if (any(!is.finite(as.matrix(results[contributionNames])))) {
    stop("domain contributions must be finite", call. = FALSE)
  }

  # Aggregate SHAP contributions at the observation level
  # ============================================================
  # Sum signed contributions within each domain before taking absolute values.

  domainRow <- stats::aggregate(
    results[, contributionNames, drop = FALSE],
    by = list(domain = results$domain, index = results$index),
    FUN = sum
  )

  # Aggregate absolute domain contributions across observations
  # ============================================================
  domainColumn <- stats::aggregate(
    abs(domainRow[, contributionNames, drop = FALSE]),
    by = list(domain = domainRow$domain),
    FUN = sum
  )

  # Preserve the user-specified order of domains
  domainColumn <- domainColumn[match(names(domains), domainColumn$domain), , drop = FALSE]
  rownames(domainColumn) <- NULL

  # Compute domain contribution ratios for each model
  # ============================================================
  contributionMatrix <- as.matrix(domainColumn[, contributionNames, drop = FALSE])
  TOTAL <- colSums(contributionMatrix)
  ratioMatrix <- contributionMatrix

  for (j in seq_along(TOTAL)) {
    if (TOTAL[j] > 0) ratioMatrix[, j] <- contributionMatrix[, j] / TOTAL[j]
    else ratioMatrix[, j] <- 0
  }

  domainRatio <- data.frame(domain = domainColumn$domain, ratioMatrix, check.names = FALSE)

  # Prepare and align model-performance weights
  # ============================================================
  original_weights <- shapley$weights
  w <- as.numeric(original_weights)
  weight_names <- names(original_weights)

  if (!is.null(weight_names)) {
    model_ids <- sub("^contribution_?", "", contributionNames)

    if (all(model_ids %in% weight_names)) {
      w <- as.numeric(original_weights[model_ids])
    } else if (all(contributionNames %in% weight_names)) {
      w <- as.numeric(original_weights[contributionNames])
    }
  }

  if (length(w) != length(contributionNames) || anyNA(w) ||
      any(!is.finite(w)) || any(w < 0) || sum(w) <= 0) {
    stop("invalid or incompatible model-performance weights", call. = FALSE)
  }

  # Normalize weights to sum to the number of models
  n_models <- length(w)
  w <- w / sum(w) * n_models

  # Compute weighted domain summaries
  # ============================================================
  SUMMARY <- data.frame(domain = names(domains), mean = NA_real_, sd = NA_real_,
                        ci = NA_real_, lowerCI = NA_real_, upperCI = NA_real_)

  for (i in seq_len(nrow(SUMMARY))) {

    tmp <- as.numeric(domainRatio[i, contributionNames, drop = TRUE])
    weighted_mean <- stats::weighted.mean(tmp, w)

    if (n_models > 1L) {
      weighted_var <- sum(w * (tmp - weighted_mean)^2) / sum(w)
      weighted_sd <- sqrt(max(0, weighted_var))
      ci <- 1.96 * weighted_sd / sqrt(n_models)
    } else {
      weighted_sd <- NA_real_
      ci <- NA_real_
    }

    SUMMARY$mean[i] <- weighted_mean
    SUMMARY$sd[i] <- weighted_sd
    SUMMARY$ci[i] <- ci
    SUMMARY$lowerCI[i] <- weighted_mean - ci
    SUMMARY$upperCI[i] <- weighted_mean + ci
  }

  # Sort domains for visualization
  # ============================================================
  SUMMARY$domain <- factor(SUMMARY$domain, levels = SUMMARY$domain[order(SUMMARY$mean)])

  # Plot weighted mean domain contributions
  # ============================================================
  if (isTRUE(plot)) {

    FILLCOLOR <- if (length(domains) <= 10L) rep_len(COLORCODE, length(domains)) else COLORCODE[1L]

    Plot <- ggplot2::ggplot(SUMMARY, ggplot2::aes(x = !!as.name("domain"), y = !!as.name("mean"))) +
      ggplot2::geom_col(fill = FILLCOLOR, alpha = 0.8) +
      ggplot2::coord_flip() +
      ggplot2::ggtitle("") +
      ggplot2::xlab(paste0(xlab, "\n")) +
      ggplot2::ylab("\nWeighted Mean SHAP contributions of domains") +
      ggplot2::theme_classic() +
      ggplot2::theme(
        plot.margin = ggplot2::margin(t = -0.5, r = 0.25, b = 0.25, l = 0.25, unit = "cm")
      ) +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.05)))

    # Display descriptive between-model intervals when multiple models are available
    if (n_models > 1L) {
      Plot <- Plot +
        ggplot2::geom_errorbar(
          ggplot2::aes(ymin = !!as.name("lowerCI"), ymax = !!as.name("upperCI")),
          width = 0.2, color = "#7A004BF0", alpha = 0.75, linewidth = 0.7
        )
    }

    base::print(Plot)
  }

  if (isTRUE(print)) base::print(SUMMARY)

  return(list(domainSummary = SUMMARY, domainRatio = domainRatio, plot = Plot))
}
