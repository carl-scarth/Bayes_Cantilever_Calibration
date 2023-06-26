import numpy as np
import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from src.cantilever_beam import *  # Import cantilever model
from src.LHS_Design import transformed_LHS  # Import Latin Hypercube module
from src.sample_prior import *
# from src.Prior import Prior

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # First we need to define our space of inputs and generate our training samples
    # ------------------------------------------------------------------------------

    # Define uncertain input prior values
    # Would be better in Pandas.
    inputs = [
        ['E', 'Gaussian', 68E9, 10.0, True],
        ['P', 'Gaussian', 10.0, 10.0, True],
        ['b', 'Gaussian', 0.04, 10.0, True],
        ['d', 'Gaussian', 0.01, 10.0, True],
        ['L', 'Gaussian', 1.0, 10.0, True]
    ]
    d = len(inputs)  # Number of inputs
    N_train = 50  # Number of required training data points
    # Generate training samples
    x_train = transformed_LHS(inputs, N_train, sampler_package="scipy")
    inp_str = [inp[0] for inp in inputs] # List of input variable names

# ------------------------------------------------------------------------------
#           Run the cantilever beam model for Design of experiments
# ------------------------------------------------------------------------------
    y_train = np.empty(shape=N_train)
    for i, x_i in enumerate(x_train):
        # x is taken as L to output tip deflection
        y_train[i] = cantilever_beam(
            x=x_i[4], E=x_i[0], b=x_i[2], d=x_i[3], P=x_i[1], L=x_i[4])

# ------------------------------------------------------------------------------
#          Generate a set of points at which predictions are required
# ------------------------------------------------------------------------------

    N_pred = 2000  # Number of predictions
    x_pred = sample_prior(inputs, N_pred)  # Generate a set of random samples
    # Run the cantilever model for the prediction points (for validation)
    y_pred = np.empty(shape=N_pred)
    for i, x_i in enumerate(x_pred):
        # x is taken as L to output tip deflection
        y_pred[i] = cantilever_beam(
            x=x_i[4], E=x_i[0], b=x_i[2], d=x_i[3], P=x_i[1], L=x_i[4])

# ------------------------------------------------------------------------------
#                        Plot the Design of Experiments
# ------------------------------------------------------------------------------
    
    N_grades = 8 # Number of different values to divide the output into
    # Create data frame as Seaborn only works with Pandas
    plot_frame = pd.concat((pd.DataFrame(x_train,columns=inp_str),pd.DataFrame(y_train,columns=["Displacement"])),axis=1)
    # Bin the displacement into intervals, and create a new categorical column
    # in the dataframe stating which interval each point lies within
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    # Create a pairs plot of the training data, coloured according to the 
    # displacement value of each point
    sns.pairplot(plot_frame, vars = inp_str, hue="Category", palette = sns.color_palette("CMRmap",N_grades), diag_kind=None, plot_kws=dict(s = 50))
    # Produce a pairs plot for the points at which predictions are required
    sns.pairplot(pd.DataFrame(x_pred, columns=inp_str))
    plt.show()
    # It would be nice to play with the font size options but I've spent enough time on this
    # I'd also like to spend more time investigating different colour palette 
    # options. Return to this once I've made more progress with the following 
    # code

# ------------------------------------------------------------------------------
#                           Standardise the data
# ------------------------------------------------------------------------------

# Plot Design of experiments
# Transform the various inputs
# Compare against MLE?
# Compare histogram, RMSE?

