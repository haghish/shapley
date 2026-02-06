#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param domains character, name of two features to be compared with permutation test
#' @param n integer, number of permutations
#' @author E. F. Haghish
#' @return normalized numeric vector
#' @examples
#'
#' \dontrun{
#' # load the required libraries for building the base-learners and the ensemble models
#' library(h2o)            #shapley supports h2o models
#' library(autoEnsemble)   #autoEnsemble models, particularly useful under severe class imbalance
#' library(shapley)
#'
#' # initiate the h2o server
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)
#'
#' # upload data to h2o cloud
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#'
#' ### H2O provides 2 types of grid search for tuning the models, which are
#' ### AutoML and Grid. Below, I demonstrate how weighted mean shapley values
#' ### can be computed for both types.
#'
#' set.seed(10)
#'
#' #######################################################
#' ### PREPARE AutoML Grid (takes a couple of minutes)
#' #######################################################
#' # run AutoML to tune various models (GBM) for 60 seconds
#' y <- "CAPSULE"
#' prostate[,y] <- as.factor(prostate[,y])  #convert to factor for classification
#' aml <- h2o.automl(y = y, training_frame = prostate, max_runtime_secs = 120,
#'                  include_algos=c("GBM"),
#'
#'                  # this setting ensures the models are comparable for building a meta learner
#'                  seed = 2023, nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' ### call 'shapley' function to compute the weighted mean and weighted confidence intervals
#' ### of SHAP values across all trained models.
#' ### Note that the 'newdata' should be the testing dataset!
#' result <- shapley(models = aml, newdata = prostate, plot = TRUE)
#'
#' #######################################################
#' ### Significance testing of contributions of two domains (or latent factors)
#' #######################################################
#' domains = list(Demographic = c("RACE", "AGE"),
#'                Cancer = c("VOL", "PSA", "GLEASON"))
#' shapley.domain.test(result, domains = domains, n=5000)
#' }
#' @export

shapley.domain.test <- function(shapley, domains, n = 5000) {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (length(domains) != 2) stop("domain must be a vector of length 2")
  # ??? improve domain element check
  # if (!all(domain %in% names(shapley$feature_importance)))
  #   stop("domain must be a subset of the domain in the shapley object")
  if (!is.numeric(n)) stop("n must be numeric")
  if (n < 100) stop("n must be greater than or equal to 100")

  # Compute domain contributions
  # ============================================================
  dom <- shapley.domain(shapley,
                        domains,
                        print = FALSE,
                        plot = NULL)

  # Prepare the variables
  # ============================================================
  var1 <- as.numeric(dom$domainRatio[dom$domainRatio$domain == names(domains)[1],
                          grep("^contribution", names(dom$domainRatio))])
  var2 <- as.numeric(dom$domainRatio[dom$domainRatio$domain == names(domains)[2],
                          grep("^contribution", names(dom$domainRatio))])

  # var1    <- unlist(shapley$feature_importance[features[1]])
  # var2    <- unlist(shapley$feature_importance[features[2]])
  weights <- shapley$weights

  # Run the test
  # ============================================================
  print(class(var1))
  results <- feature.test(var1, var2, weights, n)

  if (results$p_value < 0.05) {
    message(paste0("The difference between the two domains is significant:\n",
                  "observed Weighted Mean Shapley (WMSHAP) difference = ", as.character(results$mean_shapley_diff), " and ",
                  "p-value = ", as.character(results$p_value)))

  } else {
    message(paste0("The difference between the two domains is not significant:\n",
                   "observed Weighted Mean Shapley (WMSHAP) difference =", as.character(results$mean_shapley_diff), " and ",
                   "p-value = ", as.character(results$p_value)))
  }

  return(results)
}

# print(shapley.domain.test(result, domains = domains, n=5000))
# shapley.test(a, features = c("AGE", "PSA"), n=5000)
