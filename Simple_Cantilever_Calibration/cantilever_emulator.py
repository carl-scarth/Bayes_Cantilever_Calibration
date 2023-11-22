# Fits a Gaussian process emulator to the point-wise displacement of a simple cantilever beam
# with uncertain depth

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from maximin import *
import transform_input_output as tio

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # First we need to define our space of inputs and generate our training samples
    # amd set up model parameters
    # ------------------------------------------------------------------------------

    E = 68E9 # Young's modulus
    P = 10.0 # Applied load
    b = 0.04 # Beam width
    L = 1.0  # Beam length
    d = 10.0 # Nominal value of  beam height d

    N_train = 25  # Number of required training data points
    # Generate training samples
    # Define uncertain input prior values
    # Would be better in Pandas.
    inputs = [
        ['d', 'Uniform', 0.9*d, 1.1*d, True],
    ]
    x_train = transformed_LHS(inputs, N_train, sampler_package="scikit-optimize", sampler_kwargs={"lhs_type":"classic","criterion":"maximin", "iterations":100})
    inp_str = [inp[0] for inp in inputs] # List of input variable names
    inp_str.append("x")

    N_plot = 100   # Number of points at which beam deflection is plotted
    N_maximin = 50 # Number of model output data points which are to be retained for training the emulator 

    # ------------------------------------------------------------------------------
    #           Run the cantilever beam model for Design of experiments
    # ------------------------------------------------------------------------------
    
    # Define a list of coordinates at which the simulation output is generated.
    sim_coords = [i/(N_plot-1)*L for i in range(N_plot)]
    
    # Run the cantilever model n_simulations times
    y_all = np.empty(shape = (N_plot,N_train))
    for i, x_i in enumerate(x_train):
        y_all[:,i] = cantilever_beam(x=sim_coords,E=E,b=b,d = x_i[0],P=P, L=L)

    # Reshape all n_simulations x N_train points into a vector
    y_all = y_all.T.reshape((-1))
    
    # Create a matrix matching each training data point (x-coordinate and depth) to each output value
    sim_coords = np.array(sim_coords,ndmin=2).T
    x_all = np.concatenate((np.repeat(x_train,N_plot,axis=0), np.tile(sim_coords,(N_train,1))), axis=1)

    # For speed we need to cherry-pick points. Choose observations to satisfy the maximin criterion
    # First transform inputs onto unit hypercube (note - we can also use scikit-learn.preprocessing)
    # Normalise inputs such that training data lies on the unit hypercube
    x_all_trans, *_ = tio.normalise_inputs(x_all)
    # not very efficient - could speed things up by optimising choice of replacement, rather than
    # selecting at random. An obvious choice would be to replace one of the points with the closest 
    # distance from it's nearest neighbour. Could also add in the corners to the training sample
    # see R maximin documentation
    x_maximin, ind, d_min = rand_maximin(x_all_trans, N_maximin, 15000)
    # Extract untransformed inputs and outputs for the maximin sample
    x_train = x_all[ind,:]
    y_train = y_all[ind]

# ------------------------------------------------------------------------------
#                        Plot the Design of Experiments
# ------------------------------------------------------------------------------
    
    # Set plot parameters
    plt.rcParams.update({'font.size': 16,
                         'figure.titlesize' : 18,
                         'axes.labelsize': 18,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'legend.fontsize': 15})
    
    N_grades = 8 # Number of different values to divide the output into
    
    # Create data frame as Seaborn only works with Pandas
    plot_frame = pd.concat((pd.DataFrame(x_train,columns=inp_str),pd.DataFrame(y_train,columns=["Displacement"])),axis=1)
    # Bin the displacement into intervals, and create a new categorical column
    # in the dataframe stating which interval each point lies within
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    # Create a pairs plot of the training data, coloured according to the 
    # displacement value of each point
    fig, axes = plt.subplots(1,2, sharey = True)
    fig.suptitle("Training data before and after maximin search")
    # There are only two inputs so it is more appropriate to use scatterplot thann pairplot
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes[1], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))
    plot_frame = pd.concat((pd.DataFrame(x_all, columns = inp_str),pd.DataFrame(y_all,columns=["Displacement"])),axis=1)
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes[0], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))

# ------------------------------------------------------------------------------
#          Generate a set of points at which predictions are required
# ------------------------------------------------------------------------------

    # Generate a sequence of points across x and t
    n_grid = 15
    x_grid = np.linspace(min(sim_coords),max(sim_coords), num = n_grid).reshape(-1)
    t_grid = np.linspace(0.9*d,1.1*d, num = n_grid)

    # Take the union of these points with the training data - we want to verify
    # that uncertainty is zero at these points
    x_grid = np.sort(np.unique(np.concatenate((x_grid,x_train[:,1]))))
    t_grid = np.sort(np.unique(np.concatenate((t_grid,x_train[:,0]))))

    # Expand points into a grid
    x_grid, t_grid = np.meshgrid(x_grid,t_grid)
    x_pred = np.concatenate((x_grid.reshape(-1,1), t_grid.reshape(-1,1)),axis=1)
    print(x_pred)
    asdsads

    # Keep working copying and pasting code from both R and tip exercise
    # Consider improving maximin code if at a loose end

#-----------------------------------------------------------------------------

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------

    # Will this be calculated from combined matrix with coordinate included?
 
#   d = len(inputs)  # Number of inputs If using call something else as d defined for nominal depth

    #-------------------------------------------------------------------------------

# When done, convert to Pandas and see if pymc behaves any differently
# Continue working with 
# what about numpyro?
# Experiment to see if pymc any quick than stan for large N_maximin



