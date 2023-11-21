from pyexpat.errors import XML_ERROR_ENTITY_DECLARED_IN_PE
import numpy as np

# Code for determining a random subset of points from a pandas dataframe which maximises the minimum distace between points
# Should be easyto modify for pandas using df.sample()
def rand_maximin(x, N_sam, N_iter):
    # Take a random subset of length N_sam for dataframe x, then
    # iteratively add random points if they increase the minimum distance 
    # between neighbouring points, reject these otherwise
    ind = np.arange(x.shape[0])
    subset_ind = np.random.choice(ind, N_sam, replace = False)
    x_subset = x[subset_ind]
    # calculate the minimum distance between two rows in the array
    d_min = np.min(nn_dist(x_subset))
    replace_ind = np.arange(N_sam)
    for i in range(N_iter):
        # Randomly replace a row in x_subset with a row from x
        i_cand = np.random.choice(np.delete(ind, subset_ind))
        x_cand = x[i_cand,:]
        i_rep = np.random.choice(replace_ind)
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

    return(x_subset, subset_ind, d_min)

def nn_dist(x):
    # Calculate the squared Euclidean distance to the nearest neighbour for all rows in a numpy array
    N = x.shape[0]
    d_nn = np.empty(N)
    for i, x_i in enumerate(x):
        d_nn[i] = np.min(np.sum(np.square(np.delete(x,i,axis=0) - x_i),axis=1))

    return(d_nn)