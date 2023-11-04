#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param features character, name of two features to be compared with permutation test
#' @param n integer, number of permutations
#' @author E. F. Haghish
#' @return normalized numeric vector
#' @export

shapley.test <- function(shapley, features, n = 5000) {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (length(features) != 2) stop("features must be a vector of length 2")
  if (!all(features %in% names(shapley$feature_importance)))
    stop("features must be a subset of the features in the shapley object")
  if (!is.numeric(n)) stop("n must be numeric")
  if (n < 100) stop("n must be greater than or equal to 100")

  # Prepare the variables
  # ============================================================
  var1    <- unlist(shapley$feature_importance[features[1]])
  var2    <- unlist(shapley$feature_importance[features[2]])
  weights <- shapley$weights

  # Run the test
  # ============================================================
  results <- test(var1, var2, weights, n)

  if (results$p_value < 0.05) {
    message(paste0("The difference between the two features is significant:\n",
                  "observed weighted mean Shapley difference = ", as.character(results$mean_shapley_diff), " and ",
                  "p-value = ", as.character(results$p_value)))

  } else {
    message(paste0("The difference between the two features is not significant:\n",
                   "observed weighted mean Shapley difference =", as.character(results$mean_shapley_diff), " and ",
                   "p-value = ", as.character(results$p_value)))
  }
  print(results)

  return(results)
}

# shapley.test(a, features = c("AGE", "PSA"), n=5000)
