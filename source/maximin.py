import numpy as np
import pandas as pd

# Code for determining a random subset of points from a pandas dataframe which maximises the minimum distace between points
# Should be easy to modify for pandas using df.sample()
def rand_maximin(x, N_sam, N_iter, N_init = [], replace_method = "random"):
    # If not already, convert data to the correct format, or raise exception
    if isinstance(x, pd.DataFrame):
        print("pandas dataframe")
        x = x.to_numpy()
    elif not isinstance(x, np.ndarray):
        raise Exception("Please input either a numpy array or pandas DataFrame")
    
    # Use default value for number of initialisation steps if not specified
    if len(N_init) < 1:
        N_init = min([x.shape[1]*15, 100])
    
    # Take a random subset of length N_sam for dataframe x, then
    # iteratively add random points if they increase the minimum distance 
    # between neighbouring points, reject these otherwise. Take N_init random
    # initial subsets and pick the best one
    x_subset, subset_ind, d_min = init_subset(x, N_sam, N_init)
        
    ind = np.arange(x.shape[0])
    replace_ind = np.arange(N_sam)
    # If picking cadidate replacements according to minimum distance, we need
    # the argmin for the initial candidate set
    if replace_method == "min":
        d_argmin = np.argmin(nn_dist(x_subset))

    for i in range(N_iter):
        # Randomly replace a row in x_subset with a row from x
        i_cand = np.random.choice(np.delete(ind, subset_ind))
        x_cand = x[i_cand,:]
        if replace_method == "random":
            i_rep = np.random.choice(replace_ind)
        elif replace_method == "min":
            # Pick point which had closest distance from it's nearest neighbour
            # in previous iteration
            # Doesn't work as well as random, potential for being trapping in
            # local optima
            i_rep = d_argmin
        else:
            raise Exception("Please enter a valid replacement method")
        x_new = np.copy(x_subset)
        x_new[i_rep,:] = x_cand
        # Calculate minimum distance between two rows of the new subset
        d_new = np.min(nn_dist(x_new))
        # If the current minimum distance is less than the previous one,
        # update the subset
        if d_new > d_min:
            x_subset = x_new
            d_min = d_new
            subset_ind[i_rep] = i_cand
            if replace_method == "min":
                d_argmin = np.argmin(nn_dist(x_new))

    return(x_subset, subset_ind, d_min)

def nn_dist(x):
    # Calculate the squared Euclidean distance to the nearest neighbour for all rows in a numpy array
    N = x.shape[0]
    d_nn = np.empty(N)
    for i, x_i in enumerate(x):
        d_nn[i] = np.min(np.sum(np.square(np.delete(x,i,axis=0) - x_i),axis=1))

    return(d_nn)

# Take a set of N_init random subsets from x, and pick the one with the highest
# maximin criterion
def init_subset(x, N_sam, N_init):
    d_min = 0.0
    for i in range(N_init):
        ind = np.arange(x.shape[0])
        # Take a number of subsets and pick the one with the highest maximin
        new_ind = np.random.choice(ind, N_sam, replace = False)
        # x_subset = x[subset_ind]
        x_new = x[new_ind]
        # calculate the minimum distance between two rows in the array
        d_new = np.min(nn_dist(x_new))
        if d_new > d_min:
            x_subset = x_new
            subset_ind = new_ind
            d_min = d_new
    
    return(x_subset, subset_ind, d_min)