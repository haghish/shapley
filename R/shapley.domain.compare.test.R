#' @title Independent weighted bootstrap test for the difference between two domain-level WMSHAP means
#' @description Compares the weighted mean contribution of a specific domain between
#'              two \code{"shapley"} objects.
#'              The observed effect is defined as the difference between the weighted mean
#'              domain contributions in the two shapley objects.
#' @param shapley1 Object of class \code{"shapley"}.
#' @param shapley2 Object of class \code{"shapley"}.
#' @param domains Named list of domains passed to \code{shapley.domain()}. The list must
#'                contain at least 2 domains.
#' @param domain Character string naming the domain to be compared between the two
#'               \code{"shapley"} objects.
#' @param n Integer, number of bootstrap resamples used for the confidence interval and
#'          p-value calculation. Must be at least 100.
#' @param conf.level Numeric scalar between 0 and 1 giving the confidence level for the
#'                   bootstrap confidence interval.
#' @param report logical; if TRUE (default) the results are explained.
#' @param seed Optional integer seed for reproducibility.
#' @details The function carries out an independent weighted bootstrap comparison of
#'          domain-level weighted mean SHAP (WMSHAP) values. For the selected domain,
#'          model-specific domain contribution ratios are extracted from each
#'          \code{"shapley"} object. The model weights stored in each object are then
#'          normalized to sum to 1 and used to compute the weighted mean domain
#'          contribution in each sample.
#'
#'          The observed difference is defined as the weighted mean domain contribution
#'          in \code{shapley1} minus the weighted mean domain contribution in
#'          \code{shapley2}. A nonparametric weighted bootstrap is then used to estimate
#'          the confidence interval of this difference. Within each sample, bootstrap
#'          resamples are drawn with replacement from the model-specific domain
#'          contributions, using the normalized model weights as sampling probabilities.
#'          The bootstrap distribution of the difference between the two resampled means
#'          is used to construct the confidence interval.
#'
#'          To test the null hypothesis that the two weighted mean domain contributions
#'          are equal, the function generates a pooled weighted bootstrap null
#'          distribution. The domain contributions from both samples are pooled, the
#'          corresponding weights are combined and renormalized, and bootstrap samples of
#'          the original sample sizes are repeatedly drawn from this pooled weighted
#'          distribution. The two-sided p-value is computed as the proportion of null
#'          bootstrap differences that are at least as extreme as the observed
#'          difference, using the finite-sample correction \code{(b + 1) / (n + 1)}.
#'
#'          For reporting in APA style, the following values are usually sufficient:
#'          the domain name, the weighted mean domain contribution in each sample, the
#'          difference between the two weighted means, the confidence interval, and the
#'          p-value. A typical report can be written as follows: For the
#'          \emph{[domain]} domain, the weighted mean contribution was
#'          \emph{M}\eqn{_w} = x.xx in Sample 1 and \emph{M}\eqn{_w} = x.xx in Sample 2,
#'          with a difference of \eqn{\Delta}WMSHAP = x.xx, 95\% CI [x.xx, x.xx],
#'          \emph{p} = .xxx, based on \code{n} weighted bootstrap resamples. Because this
#'          function uses bootstrap inference rather than a t test, no t statistic or
#'          degrees of freedom are reported.
#' @return A list containing the selected \code{domain}, the weighted mean domain
#'         contribution in the first sample (\code{mean_wmshap_1}), the weighted mean
#'         domain contribution in the second sample (\code{mean_wmshap_2}), the
#'         difference between the two weighted means (\code{mean_wmshap_diff}), the
#'         lower and upper confidence interval bounds (\code{ci_lower},
#'         \code{ci_upper}), and the two-sided bootstrap \code{p_value}.
#' @author E. F. Haghish
#' @export

shapley.domain.compare.test <- function(shapley1,
                                        shapley2,
                                        domains, #list of domains with length more than 1
                                        domain, #name of the domain of interest to be compared between the shapley (wmshap) data
                                        n = 2000,
                                        conf.level = 0.95,
                                        report = TRUE,
                                        seed = NULL) {


  alpha <- 1 - conf.level

  # Helper functions
  # ============================================================
  .normalize_weights <- function(w) {
    if (!is.numeric(w) || anyNA(w) || any(w < 0) || sum(w) <= 0) {
      stop("`weights` must be numeric, non-missing, non-negative, and sum to a positive value.", call. = FALSE)
    }
    w / sum(w)
  }

  .weighted_var <- function(x, w) {
    mu  <- stats::weighted.mean(x, w)
    ess <- 1 / sum(w^2)

    if (ess <= 1) {
      return(0)
    }

    sum(w * (x - mu)^2) * ess / (ess - 1)
  }

  .bootstrap_domain_mean <- function(x, prob) {
    idx <- sample.int(length(x),
                      size = length(x),
                      replace = TRUE,
                      prob = prob)
    mean(x[idx])
  }

  # Syntax check
  # ============================================================
  if (!inherits(shapley1, "shapley"))
    stop("`shapley` must be of class 'shapley'.", call. = FALSE)
  if (!inherits(shapley2, "shapley"))
    stop("`shapley` must be of class 'shapley'.", call. = FALSE)

  if (!is.list(domains) || length(domains) < 2L) {
    stop("`domains` must be a named list with at least 2 domains", call. = FALSE)
  }
  if (is.null(names(domains)) || anyNA(names(domains)) || any(names(domains) == "")) {
    stop("`domains` must be a named list with at least 2 domains.", call. = FALSE)
  }
  if (!is.character(domain) || length(domain) != 1L || is.na(domain)) {
    stop("`domain` must be a single character string.", call. = FALSE)
  }

  # make sure domain is defined within domains
  if (!domain %in% names(domains)) {
    stop("`domain` must match one of `names(domains)`.", call. = FALSE)
  }

  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 100) {
    stop("`n` must be a single number >= 100.", call. = FALSE)
  }

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1L || is.na(seed))) {
    stop("`seed` must be NULL or a single numeric value.", call. = FALSE)
  }

  n <- as.integer(n)

  # Compute domain contributions from the first shapley objects
  # ============================================================
  dom1 <- shapley.domain(shapley1, domains, print = FALSE, plot = FALSE)
  COLUMNS1 <- grep("^contribution", names(dom1$domainRatio), value = TRUE)
  if (length(COLUMNS1) == 0L) {
    stop("No contribution columns were found in `dom1$domainRatio`.", call. = FALSE)
  }

  # Compute domain contributions from the second shapley objects
  # ============================================================
  dom2 <- shapley.domain(shapley2, domains, print = FALSE, plot = FALSE)
  COLUMNS2 <- grep("^contribution", names(dom2$domainRatio), value = TRUE)
  if (length(COLUMNS2) == 0L) {
    stop("No contribution columns were found in `dom2$domainRatio`.", call. = FALSE)
  }

  # Prepare the variables
  # ============================================================
  var1 <- as.numeric(dom1$domainRatio[dom1$domainRatio$domain == domain, COLUMNS1])
  w1 <- .normalize_weights(as.numeric(shapley1$weights))
  var2 <- as.numeric(dom2$domainRatio[dom2$domainRatio$domain == domain, COLUMNS2])
  w2 <- .normalize_weights(as.numeric(shapley2$weights))

  if (length(var1) == 0L) {
    stop("`domain` was not found in `dom1$domainRatio`.", call. = FALSE)
  }

  if (length(var2) == 0L) {
    stop("`domain` was not found in `dom2$domainRatio`.", call. = FALSE)
  }

  if (length(var1) != length(w1)) {
    stop("Length mismatch between `var1` and `shapley1$weights`.",
         call. = FALSE)
  }

  if (length(var2) != length(w2)) {
    stop("Length mismatch between `var2` and `shapley2$weights`.",
         call. = FALSE)
  }

  # Run the test
  # ============================================================
  mean1 <- stats::weighted.mean(var1, w1)
  mean2 <- stats::weighted.mean(var2, w2)
  obs_diff <- mean1 - mean2

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Bootstrap CI under the observed distributions
  # ============================================================
  boot_diff <- replicate(n, {
    .bootstrap_domain_mean(var1, w1) - .bootstrap_domain_mean(var2, w2)
  })

  ci <- stats::quantile(boot_diff,
                        probs = c(alpha / 2, 1 - alpha / 2),
                        names = FALSE,
                        na.rm = TRUE,
                        type = 6)

  # Run the bootstrap test
  # ============================================================
  # Bootstrap test under the null hypothesis
  # ============================================================
  pooled_values <- c(var1, var2)

  pooled_weights <- .normalize_weights(
    c(w1 * length(var1), w2 * length(var2))
  )

  null_diff <- replicate(n, {
    idx1 <- sample.int(length(pooled_values),
                       size = length(var1),
                       replace = TRUE,
                       prob = pooled_weights)

    idx2 <- sample.int(length(pooled_values),
                       size = length(var2),
                       replace = TRUE,
                       prob = pooled_weights)

    mean(pooled_values[idx1]) - mean(pooled_values[idx2])
  })

  p_value <- (sum(abs(null_diff) >= abs(obs_diff)) + 1) /
    (n + 1)

  results <- list(
    domain = domain,
    mean_wmshap_1 = mean1,
    mean_wmshap_2 = mean2,
    mean_wmshap_diff = obs_diff,
    ci_lower = ci[1],
    ci_upper = ci[2],
    p_value = p_value
  )

  if (report) {
    if (results$p_value < 0.05) {
      sig_txt <- "The domain difference between the two samples was statistically significant."
    } else {
      sig_txt <- "The domain difference between the two samples was not statistically significant."
    }

    p_txt <- if (results$p_value < .0001) {
      "p < 0.0001"
    } else {
      paste0("p = ", sprintf("%.4f", results$p_value))
    }

    message(paste0("\nFor the '", results$domain, "' domain, the weighted mean contribution was ",
                   "M_w = ", sprintf("%.3f", results$mean_wmshap_1), " in Sample 1 and ",
                   "M_w = ", sprintf("%.3f", results$mean_wmshap_2), " in Sample 2, with a difference of ",
                   "Delta WMSHAP = ", sprintf("%.3f", results$mean_wmshap_diff), ", ",
                   round(conf.level * 100), "% CI [",
                   sprintf("%.3f", results$ci_lower), ", ",
                   sprintf("%.3f", results$ci_upper), "], ",
                   p_txt, ", based on ", n, " weighted bootstrap resamples."))
  }

  return(results)
}

