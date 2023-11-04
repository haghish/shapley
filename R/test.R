#' Weighted Permutation Test for Difference of Means
#'
#' This function performs a weighted permutation test to determine if there is a significant
#' difference between the means of two weighted numeric vectors. It tests the null hypothesis
#' that the difference in means is zero against the alternative that it is not zero.
#'
#' @param var1 A numeric vector.
#' @param var2 A numeric vector of the same length as \code{var1}.
#' @param weights A numeric vector of weights, assumed to be the same for both \code{var1} and \code{var2}.
#' @param n The number of permutations to perform (default is 1000).
#' @return A list containing the observed difference in means and the p-value of the test.
# @examples
# var1 <- rnorm(100)
# var2 <- rnorm(100)
# weights <- runif(100)
# result <- weighted_permutation_test(var1, var2, weights)
# print(result$obs_diff)
# print(result$p_value)

test <- function(var1, var2, weights, n = 2000) {
  stopifnot(length(var1) == length(var2))
  stopifnot(length(weights) == length(var1))

  # Calculate the weighted mean difference for the original data
  obs_diff <- sum(weights * (var1 - var2)) / sum(weights)

  # Combine the data and perform permutations
  combined <- c(var1, var2)
  greater_count <- 0
  for (i in 1:n) {
    permuted <- sample(combined, length(combined))
    x_perm <- permuted[1:length(var1)]
    y_perm <- permuted[(length(var1) + 1):length(combined)]
    perm_diff <- sum(weights * (x_perm - y_perm)) / sum(weights)

    if (abs(perm_diff) >= abs(obs_diff)) {
      greater_count <- greater_count + 1
    }
  }

  # Calculate the p-value
  p_value <- greater_count / n

  return(list(
    mean_shapley_diff = obs_diff,
    p_value = p_value
  ))
}

# Example of usage:
# var1 <- c(1, 2, 3, 4, 5)
# var2 <- c(1, 2, 3, 4, 6)
# weights <- c(0.1, 0.2, 0.3, 0.4, 0.5)
# result <- weighted_permutation_test(var1, var2, weights)
# print(result$obs_diff)
# print(result$p_value)
