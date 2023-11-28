# Calibration of a simple cantilver beam model with uncertain depth using synthetic data
import sys
import numpy
from scipy.stats import uniform
import matplotlib.pyplot as plt
from matplotlib import rcParams

src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from maximin import *
from Prior import *
import transform_input_output as tio

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # Define space of inputs, generate training samples, and set up model parameters
    # ------------------------------------------------------------------------------

    N_data = 10         # number of experimental data points
    N_repeats = 3       # number of repeated experimental data points
    N_train = 30        # number of emulator training data points
    N_plot = 100        # Number of points at which beam deflection is plotted
    N_maximin = 50      # Number of model output data points which are to be retained for training the emulator
  
    # Define magnitude of the synthetic noise which will be added to model outputs to get "observed" data
    sigma_e = 0.00025 # standard deviation

    # Define nominal beam properties
    E = 68E9 # Young's modulus
    P = 10.0 # Applied load
    b = 0.04 # Beam width
    L = 1.0  # Beam length
    d = 0.01 # Nominal value of  beam height d

    # Define uncertain input prior values
    # Would be better in Pandas.
    inputs = [
        ['d', 'Uniform', 0.9*d, 1.1*d, True],
    ]
    t_prior = [Prior(inp[0],inp[1],inp[2],inp[3],inp[4]) for inp in inputs]
    x_train = transformed_LHS(inputs, N_train, sampler_package="scikit-optimize", sampler_kwargs={"lhs_type":"classic","criterion":"maximin", "iterations":100})
    inp_str = [rv.name for rv in t_prior] # List of input variable names
    inp_str.append("x")
    
    # Plotting parameters
    rcParams.update({'figure.figsize' : (8,6),
                    'font.size': 16,
                    'figure.titlesize' : 18,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 15})
    
    # ------------------------------------------------------------------------------
    #                     Generate synthetic experimental data
    # ------------------------------------------------------------------------------

     
    # Generate random beam depth used to determine "true" value. 
    # This is currently commented out for reproducability, but can be uncommented 
    # if desired
    # d_data = t_prior[0].lb + (t_prior[0].ub - t_prior[0].lb)*uniform.rvs()
  
    # Pre-define deviation from nominal beam thickness based upon previously 
    # generated data (comment out if using line 65 to generate new synthetic data)
    d_data = d + 0.0005782702 # pre-define for reproducability
    print(d_data)
    sahdsadsadsadsa
    # PINCHED FROM R VVVVVV
  
    # Define coordinates at which displacements are observed
    # (not including 0 and L, assuming it is difficult to take measurements near boundary)
    #x = seq(0.05*L, 0.95*L, length.out = n_data)
    # add some repeated observations. These arise at n_repeats randomly selected coordinates from x
    #x_repeat = x[ceiling(runif(n=n_repeats,min=0,max=n_data))]
    #x = c(x,x_repeat) # code this properly in the actual version....
    #dt_coords = data.table(x)
  
    # Calculate displacement at these points, given the "true" value of depth, using the separate "cantileverBeam.R" function
    #delta = beamDeflection(x,E,b,d + as.numeric(d_data),P)
    #dt_data = cbind(dt_coords, data.frame("D" = delta))
  
    # Generate synthetic, "observed" data by adding iid noise to the "true" displacement
    #dt_data$data = dt_data$D + rnorm(nrow(dt_data), mean = 0, sd = sigma_e)
  
    # Use the below line to write observed data to a csv file if needed later
    #write.csv(dt_data, "benchmark_observations.csv", row.names = FALSE)
  
    # Use the below line to load previously generated observed data from a csv file, if needed to reproduce past results
    # dt_data = read.table("benchmark_observations.csv", header = TRUE, sep=',')