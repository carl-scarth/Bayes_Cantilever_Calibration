# Header for standardising inputs and outputs of Gaussian processes

# Function for normalising inputs onto the unit hypercube, using (scalar or 
# vector) minimum and maximum values x_min and x_max
# std indicates whether the x is a standard deviation, in which case it isn't
# necessary to subtract the minimum
def normalise_inputs(x, x_min = [], x_max = [], std = False):
  # Calcuate maximum and minimum values of data if required
  # Assumes data x is a N x d numpy array, with axis 0 being a 
  # sample of a d-dimensional input vector
  maxmin_out = False
  if len(x_min) == 0:
    maxmin_out = True
    x_min = x.min(axis = 0)
    x_max = x.max(axis = 0)
  
  #if (is.matrix(x)){
  #  N = nrow(x)
  #} else {
  #  N = 1
  #}
  if not std:
    #x = x - x_min[rep(1,N),]
    x = x - x_min
  # x_norm = x/(x_max[rep(1,N),]-x_min[rep(1,N),])
  x_norm = x/(x_max-x_min)
 
  if maxmin_out:
    return((x_norm, x_min, x_max))
  else:
    return(x_norm)