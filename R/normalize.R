#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector.
#' @param x numeric vector
#' @param min minimum value
#' @param max maximum value
#' @author E. F. Haghish
#' @return normalized numeric vector

normalize <- function(x, min=NULL, max=NULL) {                              # Create user-defined function
  if (is.null(min)) min <- min(x)
  if (is.null(max)) max <- max(x)
  return((x - min) / (max - min))                                           # Return normalized data
}
