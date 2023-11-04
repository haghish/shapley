#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param shapley object of class 'shapley', as returned by the 'shapley' function
#' @param plot character, specifying name of the plot. the default is 'bar',
#'             creating a barplot of the mean weighted feature importance alongside
#'             their confidence intervals. Other options are "pie",
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
  if (plot != "bar" & plot != "pie" & plot != "tree") {
    stop("plot must be either 'bar', 'pie', 'tree")
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

  if (plot == "tree") {
    Plot <- ggplot(data=NULL,
                   aes(area = normalized_mean,
                       fill = features,
                       label = features)) +  #legend
      geom_treemap() +
      geom_treemap_text(colour = "white", place = "centre", grow = TRUE) +
      theme(legend.position = "right") +
      labs(fill = "Feature") +
      scale_fill_discrete(name = "Feature", labels = legend)
  }

  print(Plot)

  return(Plot)
}

#print(shapley.plot(a, plot = "tree"))
