
#' @export

shapley.table <- function(wmshap,
                          method = c("shapratio"),
                          cutoff = 0.01,
                          round = 3,
                          exclude_features = NULL,
                          dict = NULL,
                          markdown.table = TRUE,
                          split.tables = 120,
                          split.cells = 50) {

  library(psych)
  library(pander)

  # Exclude features that do not meet the criteria
  # ====================================================
  summaryShaps <- wmshap$summaryShaps
  summaryShaps <- summaryShaps[summaryShaps[,method] >= cutoff, ]
  summaryShaps <- summaryShaps[!summaryShaps[,"feature"] %in% exclude_features, ]


  # Sort the results
  summaryShaps <- summaryShaps[order(summaryShaps$mean, decreasing = TRUE), ]
  summaryShaps[, c("mean", "lowerCI", "upperCI")] <- round(summaryShaps[, c("mean", "lowerCI", "upperCI")], round)
  included_features <- summaryShaps$feature
  #View(summaryShaps)


  # Prepare a table
  # ====================================================
  Confidence <- paste0(summaryShaps$lowerCI, " - ", summaryShaps$upperCI)
  summaryShaps$WMSHAP <- paste0(summaryShaps$mean, " (", Confidence, ")")
  summaryShaps <- summaryShaps[,c("feature","WMSHAP")]

  # Add item description
  # ====================================================
  if (!is.null(dict)) {
    summaryShaps$Description <- sapply(summaryShaps$feature, function(x) {
      if (x %in% dict$name) {
        dict$description[dict$name == x]
      } else {
        paste(x)
      }
    })
  }
  else summaryShaps$Description <- summaryShaps$feature



  rownames(summaryShaps) <- NULL

  # make R avoid scientific number notation

  if (markdown.table) {
    return(pander(summaryShaps[, c("Description", "WMSHAP")],
                  justify = "left",
                  split.tables = 120,
                  split.cells = 80))
  }
  else {
    return(summaryShaps[, c("Description", "WMSHAP")])
  }
}

#shapley.table(wmshap, method = "shapratio", cutoff = 0.01, dict = dictionary(raw, attribute = "label"))
#shapley.table(wmshap, method = "shapratio", cutoff = 0.01, dict = dict)
#shapley.table(wmshap, method = "shapratio", cutoff = 0.01)
