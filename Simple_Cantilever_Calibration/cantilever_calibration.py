# Calibration of a simple cantilver beam model with uncertain depth using synthetic data
import sys
import numpy as np
import pandas as pd
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from maximin import *
from Prior import *
from transform_input_output import normalise_inputs, standardise_output, rescale_output 
import utils

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
    N_train = 25  # Number of required training data points
    inputs = [
        ['d', 'Uniform', 0.9*d, 1.1*d, True],
    ]
    
    # Plotting parameters
    rcParams.update({'figure.figsize' : (8,6),
                    'font.size': 16,
                    'figure.titlesize' : 18,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 15})
    
    # Set plotting parameters
    utils.set_plot_params()
    
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
    #              Generate simulation data for training emulator
    #-------------------------------------------------------------------------------
  
    # Define a list of coordinates at which the simulation output is generated.
    sim_coords = [i/(N_plot-1)*L for i in range(N_plot)]
    
    # Generate a set of samples of uncertain input d via Latin Hypercube Sampling
    x_train = transformed_LHS(inputs, N_train, sampler_package="scikit-optimize", sampler_kwargs={"lhs_type":"classic","criterion":"maximin", "iterations":100})
    inp_str = [inp[0] for inp in inputs] # List of input variable names
    inp_str.append("x")

    # Run the cantilever model n_simulations times
    y_all_plot = np.empty(shape = (N_plot, N_train))
    for i, x_i in enumerate(x_train):
        y_all_plot[:,i] = cantilever_beam(x=sim_coords,E=E,b=b,d = x_i[0],P=P, L=L)
  
    # Reshape all n_simulations x N_train points into a vector
    y_all = y_all_plot.T.reshape((-1))
    
    # Create a matrix matching each training data point (x-coordinate and depth) to each output value
    sim_coords = np.array(sim_coords,ndmin=2).T
    x_all = np.concatenate((np.repeat(x_train,N_plot,axis=0), np.tile(sim_coords,(N_train,1))), axis=1)  

    # For speed we need to cherry-pick points. Choose observations to satisfy the maximin criterion
    # First transform inputs onto unit hypercube (note - we can also use scikit-learn.preprocessing)
    x_all_trans, *_ = normalise_inputs(x_all)
    # not very efficient - could speed things up by optimising choice of replacement, rather than
    # selecting at random. An obvious choice would be to replace one of the points with the closest 
    # distance from it's nearest neighbour. Could also add in the corners to the training sample
    # see R maximin documentation
    x_maximin, ind, d_min = maximin(x_all_trans, N_maximin, 15000, replace_method="random")
    # Extract untransformed inputs and outputs for the maximin sample
    x_train = x_all[ind,:]
    y_train = y_all[ind]
  
    # If using Pandas dataframe (haven't changed any of the above code so may be able to do this also)
    x_train = pd.DataFrame(x_train, columns = inp_str)
    y_train = pd.Series(y_train, name="Displacement")

    # ------------------------------------------------------------------------------
    #              Plot the Design of Experiments and experimental data
    #-------------------------------------------------------------------------------
    
    # First plot beam displacement 
    # Plot observed displacements
    fig, ax = plt.subplots()
      
    # define points at which the "true" displacement is to be plotted
    x_plot = [i/(N_plot-1)*L for i in range(N_plot)]
        # run beam model to determine "true" displacement at these points
    delta_plot = cantilever_beam(x=x_plot,E=E,b=b,d=d_data,P=P,L=L)
    ax.plot(sim_coords,y_all_plot[:,0],"-c", linewidth=0.5, label = "Prior simulation")
    ax.plot(sim_coords,y_all_plot[:,1:],"-c", linewidth=0.5)
    ax.plot(x_plot, delta_plot,"-b",linewidth=2, label="True response")
    ax.plot(dt_data.x.values, dt_data.delta.values,"rx",markersize=12, markeredgewidth=2, label="Experimental data")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("displacement (m)")
    ax.set_xlim([0,L])
    ax.set_title("Experimental data vs \"true\" displacement")
    ax.legend()
    
    # Plot DoE across space of inputs
    # Bin the displacement into intervals, and create a new categorical column
    # in the dataframe stating which interval each point lies within
    N_grades = 8 # Number of different values to divide the output into
    # Create data frame as Seaborn only works with Pandas
    plot_frame = pd.concat((x_train, y_train),axis=1)
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    # Create a pairs plot of the training data, coloured according to the 
    # displacement value of each point
    fig2, axes2 = plt.subplots(1,2, sharey = True)
    fig2.suptitle("Training data before and after maximin search")
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes2[1], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))
    # There are only two inputs so it is more appropriate to use scatterplot thann pairplot
    plot_frame = pd.concat((pd.DataFrame(x_all, columns = inp_str),pd.DataFrame(y_all,columns=["Displacement"])),axis=1)
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes2[0], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))

    # From here!!

    asdsadsa


    plt.show()
    # If I want
    # vvvv HOW TO CREATE NUMPY KERNEL FUNCTION FOR GP - THIS IS WHAT I NEED
    # LOOKS QUITE OLD SO MAY BE OUT OF DATE
    # https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/gaussian_processes/gaussian_process.html
    # sIMILAR
    # https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/gaussian_processes/GP-MeansAndCovs.html
    # THIS IS THE MORE UP TO DATE VERSION
    # https://www.pymc.io/projects/docs/en/v5.10.1/learn/core_notebooks/pymc_pytensor.html