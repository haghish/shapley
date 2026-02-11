#' @title Weighted permutation test for paired difference of means
#'
#' @description Performs a weighted permutation test for the null hypothesis that the
#'              weighted mean of (var1 - var2) is zero.
#'
#' @param var1 A numeric vector.
#' @param var2 A numeric vector of the same length as \code{var1}.
#' @param weights A numeric vector of non-negative weights of the same length as \code{var1} and \code{var2}.
#' @param n Integer. Number of permutations (default 2000).
#' @return A list with:
#' \describe{
#'   \item{mean_wmshap_diff}{Observed weighted mean difference (var1 - var2).}
#'   \item{p_value}{Monte Carlo permutation p-value.}
#' }
#' @examples
#' \dontrun{
#' var1 <- rnorm(100)
#' var2 <- rnorm(100)
#' weights <- runif(100)
#' result <- shapley:::feature.test(var1, var2, weights)
#' result$mean_wmshap_diff
#' result$p_value
#' }

feature.test <- function(var1, var2, weights, n = 2000) {

  # Syntax check
  # ============================================================
  if (!is.numeric(var1) || !is.numeric(var2) || !is.numeric(weights)) {
    stop("'var1', 'var2', and 'weights' must be numeric.", call. = FALSE)
  }
  stopifnot(length(var1) == length(var2))
  stopifnot(length(weights) == length(var1))
  if (anyNA(var1) || anyNA(var2) || anyNA(weights)) {
    stop("Missing values are not supported", call. = FALSE)
  }
  if (any(weights < 0)) {
    stop("'weights' must be non-negative.", call. = FALSE)
  }
  if (is.na(n) || n < 1) {
    stop("'n' must be a positive integer.", call. = FALSE)
  }

  # Variables
  # ============================================================
  # Combine the data and perform permutations
  sw <- sum(weights)
  wd <- weights * (var1 - var2)

  combined <- c(var1, var2)
  weights_combined <- c(weights, weights)
  LENGTH <- length(var1)
  greater_count <- 0L

  # Calculate the weighted mean difference for the original data (paired design)
  # ============================================================
  obs_diff <- sum(weights * (var1 - var2)) / sum(weights)

  for (i in 1:n) {
    # paired permutation = random swap within each pair = sign flip of (var1 - var2)
    idx <- sample.int(2L, LENGTH, replace = TRUE)
    idx <- idx * 2L - 3L  # 1->-1, 2->+1
    perm_diff <- as.numeric(crossprod(idx, wd)) / sw

    if (abs(perm_diff) >= abs(obs_diff)) {
      greater_count <- greater_count + 1L
    }
  }

  # Monte Carlo p-value (+1 correction, recommended for random sampling permutations)
  p_value <- (greater_count + 1) / (n + 1)

  return(list(
    mean_wmshap_diff = obs_diff,
    p_value = p_value
  ))
}

# var1 <- rnorm(100)
# var2 <- rnorm(100)
# weights <- runif(100)
# result <- shapley:::feature.test(var1, var2, weights)
# result$mean_wmshap_diff
# result$p_value
