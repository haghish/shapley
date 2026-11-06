#' @title Jitter SHAP Summary Points According to Contribution Density
#'
#' @description Spreads points along each feature's axis in proportion to
#'   the local density of its signed WMSHAP contributions, creating the
#'   compact summary display used by H2O-style SHAP plots. This position
#'   adjustment does not alter WMSHAP contributions or the plotting theme.
#'
#' @param width Maximum horizontal jitter (before rotating the plot).
#' @return A ggplot2 position adjustment.
#' @noRd

shapley.jitter <- function(width = 0.4) {

  PositionWMSHAPJitter <- ggplot2::ggproto(
    "PositionWMSHAPJitter", ggplot2::Position,
    required_aes = c("x", "y"),

    setup_params = function(self, data) {
      list(width = self$width)
    },

    compute_panel = function(data, params, scales) {
      if (nrow(data) == 0L || params$width <= 0) return(data)

      for (feature in unique(data$x)) {
        rows <- which(data$x == feature & is.finite(data$y))
        if (length(rows) < 2L) next

        contributions <- data$y[rows]
        density_weight <- rep(1, length(rows))

        if (length(rows) >= 10L && diff(range(contributions)) > 0) {
          estimate <- stats::density(contributions, n = 256L)
          density_weight <- stats::approx(
            estimate$x, estimate$y, xout = contributions,
            rule = 2
          )$y
          density_weight <- density_weight / max(density_weight)
        }

        data$x[rows] <- data$x[rows] +
          stats::runif(length(rows), -params$width, params$width) * density_weight
      }

      return(data)
    }
  )

  return(ggplot2::ggproto(NULL, PositionWMSHAPJitter, width = width))
}
