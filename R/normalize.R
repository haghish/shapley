#' @title Normalize a vector based on specified minimum and maximum values
#' @description This function normalizes a vector based on specified minimum
#'              and maximum values. If the minimum and maximum values are not
#'              specified, the function will use the minimum and maximum values
#'              of the vector (ignoring missing values).
#' @param x numeric vector
#' @param min minimum value
#' @param max maximum value
#' @author E. F. Haghish
#' @return A numeric vector of the same length as \code{x}
#' @examples
#' normalize(c(0, 5, 10))
#' normalize(c(1, 1, 1))
#' normalize(c(NA, 2, 3))

normalize <- function(x, min=NULL, max=NULL) {

  # Syntax check
  # ============================================================
  if (!is.numeric(x)) {
    stop("'x' must be numeric.", call. = FALSE)
  }
  if (length(x) == 0L) return(x)
  if (is.null(min)) min <- min(x, na.rm = TRUE)
  if (is.null(max)) max <- max(x, na.rm = TRUE)
  if (min == max) stop("Cannot compute 'min' & 'max' from 'x'", call. = FALSE)

  # Normalize x
  # ============================================================
  return((x - min) / (max - min))                                           # Return normalized data
}
