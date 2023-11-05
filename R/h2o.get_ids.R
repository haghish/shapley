#' @title h2o.get_ids
#' @description extracts the model IDs from H2O AutoML object or H2O grid
#' @importFrom h2o h2o.get_leaderboard
#' @param automl a h2o \code{"AutoML"} grid object
#' @return a character vector of trained models' names (IDs)
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
#' ids <- h2o.ids(aml)
#' }
#' @export

h2o.get_ids <- function(automl) {
  if (inherits(automl,"H2OAutoML")) return(as.vector(h2o.get_leaderboard(automl)[,1]))
  else if (inherits(automl,"H2OGrid")) return(as.vector(unlist(automl@model_ids)))
}
