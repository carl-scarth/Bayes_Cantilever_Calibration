# Calibration of a simple cantilver beam model with uncertain depth using synthetic data
import sys
import numpy as np
import pandas as pd
from scipy.stats import uniform, norm
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
  
    # Define coordinates at which displacements are observed (not including 0 and 
    # L, assuming it is difficult to take measurements near boundary)
    x = [i/(N_data-1)*0.98*L + 0.01 for i in range(N_data)]

    # Add some repeated observations
    x_repeat = [x[i] for i in np.random.randint(N_data, size = N_repeats).tolist()]
    x = x + x_repeat
  
    # Calculate displacement at these points, given the "true" beam depth
    delta = cantilever_beam(x=x,E=E,b=b,d = d_data,P=P, L=L)
    dt_data = pd.DataFrame({"x": x, "delta" : delta})
  
    # Adding synthetic iid Gaussian noise to the "true" displacement
    dt_data.delta = dt_data.delta + norm.rvs(loc=0, scale = sigma_e, size = N_data+N_repeats)
    
    # ------------------------------------------------------------------------------