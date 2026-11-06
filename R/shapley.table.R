#' @title Create WMSHAP Summary Table Based on the Given Criterion
#' @description
#' #' Generates a summary table of weighted mean SHAP ratios (WMSHAP) and confidence intervals
#' for each feature based on a weighted SHAP analysis. The function filters the SHAP summary
#' table (from a \code{shapley} object) by selecting features that meet or exceed a specified
#' cutoff using a selection method (default \code{"mean"}).
#'
#' The output is sorted by WMSHAP and formatted as either a markdown table (via \pkg{pander})
#' or a data frame.
#'
#' @param shapley             A \code{shapley} object returned by \code{\link{shapley}} that contains
#'                           a data frame \code{summaryShaps}.
#' @param method             Character. The column name in \code{summaryShaps} used
#'                           for feature selection. Default is \code{"mean"}, which
#'                           selects important features which have weighted mean shap
#'                           ratio (WMSHAP) higher than the specified cutoff. Other
#'                           alternative is "lowerCI", which selects features which
#'                           their lower bound of confidence interval is higher than
#'                           the cutoff.
#' @param cutoff             Numeric. The threshold cutoff for the selection method;
#'                           only features with a value in the \code{method} column
#'                           greater than or equal to this value are retained.
#'                           Default is \code{0.01}.
#' @param round              Integer. The number of decimal places to round the
#'                           SHAP mean and confidence interval values. Default is
#'                           \code{3}.
#' @param exclude_features   Character vector. A vector of feature names to be
#'                           excluded from the summary table. Default is \code{NULL}.
#' @param dict               A data frame containing at least two columns named
#'                           \code{"name"} and \code{"description"}. If provided, the
#'                           function uses this dictionary to add human-readable feature
#'                           descriptions. Default is \code{NULL}.
#' @param markdown.table     Logical. If \code{TRUE}, the output is formatted as a
#'                           markdown table using the \pkg{pander} package; otherwise, a
#'                           data frame is returned. Default is \code{TRUE}.
#' @param split.tables       Integer. Controls table splitting in \code{pander()}.
#'                           Default is \code{120}.
#' @param split.cells        Integer. Controls cell splitting in \code{pander()}.
#'                           Default is \code{50}.
#' @param interval Interval type to display: \code{"auto"}, \code{"bmv"},
#'   \code{"bootstrap"}, and \code{"between_model"}.
#'
#' @return If \code{markdown.table = TRUE}, returns a markdown table (invisibly)
#'         showing two columns: \code{"Description"} and \code{"WMSHAP"}. If
#'         \code{markdown.table = FALSE}, returns a data frame with these columns.
#'
#  @details
#    The function works as follows:
#    \enumerate{
#      \item Filters the \code{summaryShaps} data frame from the \code{shapley}
#            object to retain only those features for which the value in the
#            \code{method} column is greater than or equal to the \code{cutoff}.
#      \item Excludes any features specified in \code{exclude_features}.
#      \item Sorts the filtered data frame in descending order by the \code{mean}
#            SHAP value.
#      \item Rounds the \code{mean}, \code{lowerCI}, and \code{upperCI} columns to
#            the specified number of decimal places.
#      \item Constructs a new \code{WMSHAP} column by concatenating the mean value
#            with its confidence interval.
#      \item Adds a \code{Description} column using the provided \code{dict} if available;
#            otherwise, uses the feature name.
#      \item Returns the final table either as a markdown table (via \pkg{pander}) or
#            as a data frame.
#    }
#'
#' @examples
#' \dontrun{
#' # load the required libraries for building the base-learners and the ensemble models
#' library(h2o)            #shapley supports h2o models
#' library(shapley)
#'
#' # initiate the h2o server
#' h2o.init(ignore_config = TRUE, nthreads = 2, bind_to_localhost = FALSE, insecure = TRUE)
#'
#' # upload data to h2o cloud
#' prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
#' prostate <- h2o.importFile(path = prostate_path, header = TRUE)
#'
#' set.seed(10)
#'
#' ### H2O provides 2 types of grid search for tuning the models, which are
#' ### AutoML and Grid. Below, I demonstrate how weighted mean shapley values
#' ### can be computed for both types.
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
#' result <- shapley(models = aml, newdata = prostate, performance_metric = "aucpr", plot = TRUE)
#'
#' #######################################################
#' ### PREPARE H2O Grid (takes a couple of minutes)
#' #######################################################
#' # make sure equal number of "nfolds" is specified for different grids
#' grid <- h2o.grid(algorithm = "gbm", y = y, training_frame = prostate,
#'                  hyper_params = list(ntrees = seq(1,50,1)),
#'                  grid_id = "ensemble_grid",
#'
#'                  # this setting ensures the models are comparable for building a meta learner
#'                  seed = 2023, fold_assignment = "Modulo", nfolds = 10,
#'                  keep_cross_validation_predictions = TRUE)
#'
#' result2 <- shapley(models = grid, newdata = prostate, performance_metric = "aucpr", plot = TRUE)
#'
#' # get the output as a Markdown table:
#' md_table <- shapley.table(shapley = result2,
#'                           method = "mean",
#'                           cutoff = 0.01,
#'                           round = 3,
#'                           markdown.table = TRUE)
#' head(md_table)
#' }
#'
#' @importFrom pander pander
#' @export
#' @author E. F. Haghish



shapley.table <- function(shapley,
                          method = "mean",
                          cutoff = 0.01,
                          round = 3,
                          exclude_features = NULL,
                          dict = NULL,
                          markdown.table = TRUE,
                          split.tables = 120,
                          split.cells = 50,
                          interval = "auto") {

  # Prepare the summary table and determine the interval type
  # ============================================================
  summaryShaps <- shapley$summaryShaps
  interval <- match.arg(interval, c("auto", "bmv", "bootstrap", "between_model", "both", "all"))

  has_bootstrap <- all(c("bootstrap_lowerCI", "bootstrap_upperCI") %in% names(summaryShaps))
  has_bmv <- all(c("bmv_lower", "bmv_upper") %in% names(summaryShaps))

  if (interval == "auto") interval <- if (has_bmv) "bmv" else "between_model"

  if (interval %in% c("bmv", "all") && !has_bmv) {
    stop("BMV is unavailable. Enable bootstrapping in shapley().", call. = FALSE)
  }

  if (interval %in% c("bootstrap", "both", "all") && !has_bootstrap) {
    stop("Bootstrap intervals are unavailable. Enable bootstrapping in shapley().", call. = FALSE)
  }

  # Select features according to the specified criterion
  # ============================================================
  if (!method %in% c("mean", "lowerCI", "bootstrap_lowerCI", "bmv_lower") ||
      !method %in% names(summaryShaps)) {
    stop("Invalid or unavailable feature-selection method", call. = FALSE)
  }

  keep <- !is.na(summaryShaps[[method]]) & summaryShaps[[method]] >= cutoff

  summaryShaps <- summaryShaps[
    keep & !summaryShaps$feature %in% exclude_features, , drop = FALSE
  ]

  summaryShaps <- summaryShaps[order(summaryShaps$mean, decreasing = TRUE), , drop = FALSE]

  # Format WMSHAP estimates and intervals
  # ============================================================
  fmt <- function(x) format(base::round(x, round), nsmall = round, trim = TRUE)
  fmt_interval <- function(lower, upper) paste0(fmt(lower), " - ", fmt(upper))

  point <- fmt(summaryShaps$mean)

  # Add feature descriptions
  # ============================================================
  descriptions <- as.character(summaryShaps$feature)

  if (!is.null(dict)) {

    if (!all(c("name", "description") %in% names(dict))) {
      stop("'dict' must contain columns 'name' and 'description'", call. = FALSE)
    }

    position <- match(descriptions, dict$name)
    found <- !is.na(position)
    descriptions[found] <- as.character(dict$description[position[found]])
  }

  # Construct the final table
  # ============================================================
  output <- data.frame(Description = descriptions, WMSHAP = point, check.names = FALSE)

  if (interval == "between_model") {

    # Preserve the original two-column output when no bootstrap is requested.
    output$WMSHAP <- paste0(
      point, " (", fmt_interval(summaryShaps$lowerCI, summaryShaps$upperCI), ")"
    )

  } else {

    if (interval %in% c("bmv", "both", "all") && has_bmv) {
      output[["BMV interval"]] <- fmt_interval(summaryShaps$bmv_lower, summaryShaps$bmv_upper)
    }

    if (interval %in% c("bootstrap", "both", "all")) {
      output[["Bootstrap 95% CI"]] <- fmt_interval(
        summaryShaps$bootstrap_lowerCI, summaryShaps$bootstrap_upperCI
      )
    }

    if (interval == "all" || (interval == "both" && !has_bmv)) {
      output[["Original between-model interval"]] <- fmt_interval(
        summaryShaps$lowerCI, summaryShaps$upperCI
      )
    }
  }

  rownames(output) <- NULL

  # Return a markdown table or data frame
  # ============================================================
  if (markdown.table) {
    return(pander::pander(output, justify = "left",
                          split.tables = split.tables, split.cells = split.cells))
  }

  return(output)
}



#shapley.table(shapley, method = "mean", cutoff = 0.01, dict = dictionary(raw, attribute = "label"))
#shapley.table(shapley, method = "mean", cutoff = 0.01, dict = dict)
#shapley.table(shapley, method = "mean", cutoff = 0.01)
