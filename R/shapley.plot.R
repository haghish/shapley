#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param plot character, specifying name of the plot. the default is 'bar',
#'             creating a barplot of the mean weighted feature importance alongside
#'             their confidence intervals. Other options are "pie", "tree", and
#'             "waffle".
#' @importFrom ggplot2 ggplot aes geom_col geom_errorbar coord_flip ggtitle xlab
#'             ylab theme_classic theme scale_y_continuous margin expansion
#'             geom_treemap
#' @importFrom treemapify geom_treemap geom_treemap_text
#' @author E. F. Haghish
#' @return ggplot2 plot
#' @export

shapley.plot <- function(shapley, plot = "bar") {

  # Syntax check
  # ============================================================
  if (!inherits(shapley, "shapley"))
    stop("shapley object must be of class 'shapley'")
  if (!is.character(plot)) {
    stop("plot must be a character string")
  }
  if (plot != "bar" & plot != "pie" & plot != "tree" & plot != "waffle") {
    stop("plot must be either 'bar', 'pie', 'tree', or 'waffle'")
  }

  index <- order(- shapley$summaryShaps$normalized_mean)
  features <- shapley$summaryShaps$feature[index]
  normalized_mean <- shapley$summaryShaps$normalized_mean[index]
  shapratio <- shapley$summaryShaps$shapratio[index]

  # Print the bar plot
  # ============================================================
  if (plot == "bar") {
    Plot <- shapley$plot
  }

  # Calculate the percentages
  percentage <- round((normalized_mean / sum(normalized_mean) * 100), 2)

  round_to_half <- function(x) {
    return(round(x * 2) / 2)
  }
  shapratio <- round_to_half(shapratio*400)

  # Create a factor with the percentage for the legend
  legend <- paste0(features, " (", percentage, "%)")
  # Order the legend by descending percentage
  #legend <- factor(legend, levels = legend[order(-percentage)])

  # Pie chart
  # ============================================================
  if (plot == "pie") {
    # Create the pie chart
    Plot <- ggplot(data=NULL, aes(x = "",
                                  y = normalized_mean, fill = legend)) +
      geom_bar(width = 1, stat = "identity") +
      coord_polar("y", start = 0) +
      theme_void() +
      theme(
        legend.title = element_blank(),
        legend.position = "right",
        text = element_text(size = 12) # Adjust text size here if needed
      ) +
      scale_fill_discrete(name = "Feature", labels = legend) +
      labs(fill = "Feature")
  }

  else if (plot == "tree") {
    Plot <- ggplot(data=NULL,
                   aes(area = normalized_mean,
                       fill = features,
                       label = features)) +  #legend
      geom_treemap() +
      geom_treemap_text(
        #colour = "white",
        place = "centre",
        grow = TRUE, reflow = TRUE, size = 1) + # Fixed font size instead of letting it be determized automatically
      theme(legend.position = "right") +
      labs(fill = "Feature") +
      scale_fill_discrete(name = "Feature", labels = legend)
  }
  else if (plot == "waffle") {
    names(shapratio) <- as.character(legend)

    # Create the waffle plot
    Plot <- waffle(shapratio, rows = 20, size = 1,
                          title = "Weighted mean SHAP contributions", legend_pos = "right")
  }

  print(Plot)

  return(Plot)
}

print(shapley.plot(a, plot = "waffle"))
