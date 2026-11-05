
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
  COLORCODE <- c("#855C75FF", "#D9AF6BFF", "#AF6458FF", "#736F4CFF",
                 "#526A83FF", "#625377FF", "#68855CFF", "#9C9C5EFF",
                 "#A06177FF", "#8C785DFF", "#467378FF", "#7C7C7CFF")

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
      weighted_var <- sum(w * (tmp - weighted_mean)^2) / (sum(w) - 1)
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

    Plot <- ggplot2::ggplot(SUMMARY, ggplot2::aes(x = domain, y = mean)) +
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

    # Display between-model intervals when multiple models are available
    if (n_models > 1L) {
      Plot <- Plot +
        ggplot2::geom_errorbar(
          ggplot2::aes(ymin = lowerCI, ymax = upperCI),
          width = 0.2, color = "#7A004BF0", alpha = 0.75, linewidth = 0.7
        )
    }

    base::print(Plot)
  }

  if (isTRUE(print)) base::print(SUMMARY)

  return(list(domainSummary = SUMMARY, domainRatio = domainRatio, plot = Plot))
}
