import numpy as np
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
  
  if not std:
    x = x - x_min
  x_norm = x/(x_max-x_min)
 
  if maxmin_out:
    return((x_norm, x_min, x_max))
  else:
    return(x_norm)
  
# Function for standardising output using mean mu_y and standard deviation
# sigma_y to have unit mean and zero standard deviation.
# std indicates whether y is a standard deviation, in which case it isn't
# necessary to centre before scaling
# There is an option to provide the mean vector and standard deviation, 
# otherwise these are calculated internally
def standardise_output(y, mu_y = [], sigma_y = [], std = False):
  if len(mu_y) < 1 or len(sigma_y) < 1:
    musd_out = True
  else:
    musd_out = False

  if not std:
    if len(mu_y) < 1:
      mu_y = np.mean(y)
    
    y = y - mu_y

  if len(sigma_y) < 1:
    sigma_y = np.std(y)

  y_scale = y/sigma_y
  if musd_out:
    return((y_scale, mu_y, sigma_y))
  else:
    return(y_scale)
  