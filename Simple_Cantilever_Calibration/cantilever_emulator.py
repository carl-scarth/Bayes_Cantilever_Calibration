# Fits a Gaussian process emulator to the point-wise displacement of a simple cantilever beam

from LHS_Design import transformed_LHS  # Import Latin Hypercube module

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # First we need to define our space of inputs and generate our training samples
    # amd set up model parameters
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

    N_train = 60  # Number of required training data points
    # Generate training samples
    x_train = transformed_LHS(inputs, N_train, sampler_package="scikit-optimize", sampler_kwargs={"lhs_type":"classic","criterion":"maximin", "iterations":10000})
    inp_str = [inp[0] for inp in inputs] # List of input variable names
    
    N_plot = 100         # Number of points at which beam deflection is plotted
    N_maximin = 60       # Number of model output data points which are to be retained for training the emulator

    # ------------------------------------------------------------------------------
    #           Run the cantilever beam model for Design of experiments
    # ------------------------------------------------------------------------------
    
    # WORK FROM HERE! THIS IS ALL COPIED FROM R

# Here I sample across the 1D depth variable by splitting its range into equal intervals,
# then sampling at random within each interval. This is reasonable given a uniform distribution. 

# define n_simulations intervals of equal size in range [-0.1,0.1]
int_start = (0:(n_simulations-1))/n_simulations*0.2-0.1
int_end = (1:n_simulations)/n_simulations*0.2-0.1
# generate n_simulations uniform random samples
int_rnd = runif(n = n_simulations, min = 0, max = 1)
# place random points within their intervals. These are the values of d for the emulator training data
d_simulation = data.table(d*((int_end-int_start)*int_rnd + int_start))
colnames(d_simulation) = "d"

# Define the points at which the simulation output is generated. Here we take N_sim_data equally spaced points
# between x = 0 and x = L
x_sim = seq(0, L, length.out = N_sim_data)

# Given we generate simulation output at each of x_sim points for each of the n_simulations simulations,
# we have a total of n_simulations x N_sim_data data points. The below aranges all of these points into a 
# table
# replicate the simulation coordinates for each n_simulations simulation
dt_sim_coords = data.table(rep(x_sim,n_simulations))
colnames(dt_sim_coords) = "x_sim"
# initialise matrix for storing n_simulations x N_sim_data output points
dt_all_simulation = matrix(0, ncol = n_simulations, nrow = n_simulations)
# running the cantilever model n_simulations times
for (i in 1:n_simulations){
  dt_all_simulation[,i] = beamDeflection(x_sim,E,b,d + as.numeric(d_simulation[i]),P)
}
 
#COPIED PYTHON FOR THE ABOVE VVVVVV
y_train = np.empty(shape=N_train)
    for i, x_i in enumerate(x_train):
        # x is taken as L to output tip deflection
        y_train[i] = cantilever_beam(x=x_i[4], E=x_i[0], b=x_i[2], d=x_i[3], P=x_i[1], L=x_i[4])

# reshape all simulated displacements into a vector, rather than a 2d grid. These are stored as follows:
#[x_1-sim_1,...,x_N_sim_data-sim_1, x_1-sim_2, ..., x_N_sim_data-sim_N_simulations]
dt_all_simulation = as.vector(dt_all_simulation)
dt_all_simulation = data.table(dt_all_simulation)
colnames(dt_all_simulation) = "D"
# We also need to duplicate the entries in the DoE for the depth properties to match this stored output
# store in format: [Sim1 x N_sim_data, Sim2 x N_sim_data, ... , sim_N_simulations x N_N_sim_data]
# Create an index of the rows you want with duplications
idx = rep(1:n_simulations, rep(N_sim_data,n_simulations))
d_simulation = d_simulation[idx, ]

# Combine all input data into one table
XT_sim = cbind(dt_sim_coords, d_simulation)

# For speed we need to cherry-pick data points. To do this we set up a design of experiments to 
# choose observations in order to satisfy the maximin criterion in the x-d space

# transform inputs onto unit hypercube prior to undertaking the maximin design. This prevents 
# any inputs with larger magnitude dominating in the distance calculations
minx = as.matrix(colMins(as.matrix(XT_sim)))
minx = matrix(rep(minx,n_simulations*N_sim_data),ncol=nrow(minx),byrow=TRUE)
maxx = as.matrix(colMaxs(as.matrix(XT_sim)))
maxx = matrix(rep(maxx,n_simulations*N_sim_data),ncol=nrow(maxx),byrow=TRUE)
XT_trans = (XT_sim - minx)/(maxx-minx)

# Use the maximin library to determine an index of the inputs which maximise the maximin criterion
sim_ind = sort(maximin.cand(n = n_maximin, Tmax = 100*n_maximin, Xcand = as.matrix(XT_trans))$inds)
# Retain only the data points with the above index
dt_simulation = cbind(XT_sim[sim_ind,], dt_all_simulation[sim_ind])
XT_sim = XT_sim[sim_ind,]

# If desired, use the below code to write training data to csv files
# write.csv(XT_sim, "benchmark_inputs.csv", row.names = FALSE)
# write.csv(dt_simulation, "benchmark_outputs.csv", row.names = FALSE)

# If desired to reproduce past results, use the below code to read benchmark data from csv file
# XT_sim = read.table("benchmark_inputs.csv", header = TRUE, sep=',')
# dt_simulation = read.table("benchmark_outputs.csv", header = TRUE, sep=',')

#-----------------------------------------------------------------------------

    # ------------------------------------------------------------------------------

    # Will this be calculated from combined matrix with coordinate included?
    d = len(inputs)  # Number of inputs

    #-------------------------------------------------------------------------------





