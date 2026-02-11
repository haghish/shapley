#' @title Extract model IDs from H2O AutoML or Grid objects
#' @description Extracts model IDs from a \code{"H2OAutoML"} object (via the leaderboard)
#'              or from a \code{"H2OGrid"} object.
#' @importFrom h2o h2o.get_leaderboard
#' @param h2oboard An object inheriting from \code{"H2OAutoML"} or \code{"H2OGrid"}.
#' @return A character vector of trained model IDs.
#' @author E. F. Haghish
#' @examples
#'
#' \dontrun{
#' library(h2o)
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#' y <- "CAPSULE"
#' prostate[,y] <- as.factor(prostate[,y])  #convert to factor for classification
#' aml <- h2o.automl(y = y, training_frame = prostate, max_runtime_secs = 30)
#'
#' # get the model IDs
#' ids <- h2o.get_ids(aml)
#' }
#' @export

h2o.get_ids <- function(h2oboard) {
  if (inherits(h2oboard,"H2OAutoML")) return(as.vector(h2o.get_leaderboard(h2oboard)[,1]))
  else if (inherits(h2oboard,"H2OGrid")) return(as.vector(unlist(h2oboard@model_ids)))
}
