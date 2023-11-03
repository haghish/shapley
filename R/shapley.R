#' @title Evaluate SHAP Values of Selected Models
#' @description Weighted average of SHAP values of selected models
#' @importFrom utils setTxtProgressBar txtProgressBar
#' @importFrom h2o h2o.stackedEnsemble h2o.getModel h2o.auc h2o.aucpr h2o.mcc
#'             h2o.F2 h2o.mean_per_class_error h2o.giniCoef h2o.accuracy
# @importFrom h2otools h2o.get_ids
#' @importFrom curl curl
#' @param models H2O search grid or AutoML grid or a character vector of H2O model IDs.
#'               the \code{"h2o.get_ids"} function from \code{"h2otools"} can
#'               retrieve the IDs from grids.
#' @param training_frame h2o training frame (data.frame) for model training
#' @param newdata h2o frame (data.frame). the data.frame must be already uploaded
#'                on h2o server (cloud). when specified, this dataset will be used
#'                for evaluating the models. if not specified, model performance
#'                on the training dataset will be reported.
#' @param family model family. currently only \code{"binary"} classification models
#'               are supported.
#' @param strategy character. the current available strategies are \code{"search"}
#'                 (default) and \code{"top"}. The \code{"search"} strategy searches
#'                 for the best combination of top-performing diverse models
#'                 whereas the \code{"top"} strategy is more simplified and just
#'                 combines the specified of top-performing diverse models without
#'                 examining the possibility of improving the model by searching for
#'                 larger number of models that can further improve the model. generally,
#'                 the \code{"search"} strategy is preferable, unless the computation
#'                 runtime is too large and optimization is not possible.
#' @param max integer. specifies maximum number of models for each criteria to be extracted. the
#'            default value is the \code{"top_rank"} percentage for each model selection
#'            criteria.
#' @param model_selection_criteria character, specifying the performance metrics that
#'        should be taken into consideration for model selection. the default are
#'        \code{"c('auc', 'aucpr', 'mcc', 'f2')"}. other possible criteria are
#'        \code{"'f1point5', 'f3', 'f4', 'f5', 'kappa', 'mean_per_class_error', 'gini', 'accuracy'"},
#'        which are also provided by the \code{"evaluate"} function.
#' @param min_improvement numeric. specifies the minimum improvement in model
#'                        evaluation metric to qualify further optimization search.
#' @param top_rank numeric vector. specifies percentage of the top models taht
#'                 should be selected. if the strategy is \code{"search"}, the
#'                 algorithm searches for the best best combination of the models
#'                 from top ranked models to the bottom. however, if the strategy
#'                 is \code{"top"}, only the first value of the vector is used
#'                 (default value is top 1\%).
#' @param stop_rounds integer. number of stoping rounds, in case the model stops
#'                    improving
#' @param reset_stop_rounds logical. if TRUE, every time the model improves the
#'                          stopping rounds penalty is resets to 0.
#' @param stop_metric character. model stopping metric. the default is \code{"auc"},
#'                    but \code{"aucpr"} and \code{"mcc"} are also available.
#' @param seed random seed (recommended)
#' @param verbatim logical. if TRUE, it reports additional information about the
#'                 progress of the model training, particularly used for debugging.
#' @return a list including the ensemble model and the top-rank models that were
#'         used in the model
#' @author E. F. Haghish
#'
#' @export

shap <- function(models,
                 newdata = NULL,
                 sample_size = 100,
                 plot = TRUE,
                 legendstyle = "categorical"
                 # family = "binary",
                 # model_selection_criteria = c("auc","aucpr","mcc","f2"),
                 # seed = -1,
                 # verbatim = FALSE
                 #loaded = TRUE,
                 #path = NULL
) {

  library(ggplot2)
  library(grid)

  # STEP 0: prepare the models
  # ============================================================
  if (inherits(models,"H2OAutoML") | inherits(models,"H2OAutoML")) {
    ids <- h2o.get_ids(models)
  }
  else if (inherits(models,"character")) {
    ids <- models
  }

  results <- NULL
  z <- 0
  pb <- txtProgressBar(z, length(ids), style = 3)

  # STEP 1: Evaluate the models for various criteria
  # ============================================================
  # This is already done before the autoEnsemble model is built and
  # should not be repeated. The model evaluation should already exist somewhere

  # STEP 2: Extract the SHAP values
  # ============================================================
  for (i in ids) {
    z <- z + 1

    m <- h2o.shap_summary_plot(
      model = h2o.getModel(i),
      newdata = newdata,
      columns = NULL #get SHAP for all columns
      #top_n_features = 5
      #sample_size = 100
    )

    # extract shap data


    if (z == 1) {
      data <- m$data #reserve the first model's data
      results <- data[, c("Row.names", "contribution")]
      results <- results[order(results$Row.names), ]
      results$w1 <- NA #this will be the performance metric of the model
      MODEL <<- m
    }
    else {
      holder <- m$data[, c("Row.names", "contribution")]
      holder$w <- NA #this will be the performance metric of the model
      colnames(holder) <- c("Row.names", paste0("contribution", z), paste0("w", z))
      holder <- holder[order(holder$Row.names), ]
      results <- cbind(results, holder[, 2:3])
    }

    setTxtProgressBar(pb, z)
  }

  datasource    <<- data
  results <<- results

  # TEST DRIVE
  # ============================================================
  # calculate the mean contribution
  mydata <- MODEL$data
  mydata <- mydata[order(mydata$Row.names), ]
  mydata <- results[, grep("^contribution", names(results))]


  meancontribution <- rowMeans(mydata)

  mydata <<- mydata
  meancontribution <<- meancontribution

  MODEL2 <- MODEL
  MODEL2$data <- MODEL2$data[order(MODEL2$data$Row.names), ]
  MODEL2$data$contribution <- meancontribution



  # STEP 3: PLOT
  # ============================================================
  if (plot) {
    if (legendstyle == "continuous") {
      print(MODEL2 <- MODEL2 +
              xlab("Features") +
              ylab("SHAP contribution") +
              ggtitle("") +
              theme_classic() +
              labs(colour = "Normalized values") +
              theme(
                legend.position="top",
                legend.justification = "right",
                legend.title.align = 0.5,
                legend.direction = "horizontal",
                legend.margin=margin(grid::unit(0,"cm")),
                legend.text=element_text(colour="black", size=6, face="bold"),
                legend.key.height = grid::unit(0.4, "cm"),
                legend.key.width = grid::unit(0.5, "cm")
              )
      )
    }
    else if (legendstyle == "categorical") {
      print(MODEL3 <- MODEL2 +
              xlab("Features") +
              ylab("SHAP contribution") +
              ggtitle("") +
              theme_classic() +
              labs(colour = "Normalized values") +
              guides(colour = guide_legend(title.position = "top",
                                           title.hjust = 0.5,
                                           # how can I pull the legend a little towards the top and reduce empty space above it?

                                           #reduce empty space above the legend
                                           legend.margin = margin(t = -1, unit = "cm"),
                                           override.aes = list(size = 3)
              )) +
              theme(
                legend.position = "top",
                legend.justification = "right",
                legend.margin = margin(unit(0, "cm")),
                legend.text = element_text(colour = "black", size = 6, face = "bold"),
                legend.key.height = unit(0.4, "cm"),
                legend.key.width = unit(0.5, "cm")
              )
      )
    }

  }

  MODEL2 <<- MODEL2


  return(list(model = MODEL,
              ids = ids))
}
