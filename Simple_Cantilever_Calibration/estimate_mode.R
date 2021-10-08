estimate_mode <- function(x) {
  # estimates mode of distribution of samples in x
  d <- density(x)
  d$x[which.max(d$y)]
}