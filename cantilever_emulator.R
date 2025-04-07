# Get's rid of the log aspect of my cantilever problem

# Set up R

library(data.table)
library(rstan)
library(maximin)
library(matrixStats)
library(plot3D)
library(plot3Drgl)

# Set current working directory. This should be modified to match the directory
# at which the stan code and any data is stored
setwd("C:/Users/cs2361/Documents/R/Gaussian Processes")

# include functions which are called in this code
source("cantileverBeam.R")

#-------------------------------------------------------------------------------

# Setup model parameters

# First we set up the parameters
n_simulations = 30   # number of simulations
N_plot = 100         # Number of points at which beam deflection is plotted
N_sim_data = 30      # number of output points in each simulation
n_maximin = 30       # Number of model output data points which are to be retained for training the emulator. I use a smaller number here as the code runs very slowly if I use everything

P <- 10 # Applied load (N)
L <- 1 # length
E <- 68E9 # Young's modulus (Aluminium)
d <- 0.01 # depth
b <- 0.04 # breadth

#-------------------------------------------------------------------------------

# Set up simulations - here we try to represent the model output using a Gaussian process emulator
# Here we would normally do a Latin Hypercube or similar to generate emulator training data, but 
# we are only sampling one variable (depth) and so this is trivial. I assume
# we have limited control over the value of x at which we get output - an FE 
# model with mesh generated using a Latin Hypercube would look pretty weird, 
# and probably give bad output

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

# plot Design of Experiments used to generate training data
dev.new(noRStudioGD = TRUE) # generate plot in new window
plot(XT_sim$x_sim,XT_sim$d,"p",main="Design of Experiments", 
     xlab = "x", ylab = "thickness deviation", xlim=c(0,L), 
     ylim=c(-0.1*d,0.1*d), pch=4, col="blue")

# plot legend
legend(x = "topright",legend = "training data point",col="blue",pch=4)

#-----------------------------------------------------------------------------

# Generate a grid of points for making emulator predictions, taken as the 
# union of points in a regular grid, and the design of experiments used to 
# train the emulator.
# It is useful to include that latter in this grid to verify that the 
# uncertainty is zero at the training data points.

# Generate a sequence of points across x and d (named t for consistency with)
# calibration code.
x_grid <- seq(0, L, length.out = 10)
t_grid <- seq(-0.1*d, 0.1*d, length.out = 10)

# Add in unique values of x and d from design of experiments to these points
x_grid = unique(c(x_grid,XT_sim$x_sim))
t_grid = unique(c(t_grid,XT_sim$d))
# Sort points in ascending order
x_grid = x_grid[order(x_grid)]
t_grid = t_grid[order(t_grid)]
# Expand points into a grid
xt <- expand.grid(x = x_grid, t = t_grid)

#-----------------------------------------------------------------------------

# Put the data into the correct format for input to stan, and run stan.

# Set up the environment for the Stan model to run in parallel. Taken from:
# https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#21_Simulating_From_A_Gaussian_Process

# set stan to execute multiple Markov chains in parallel
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
util = new.env()
par(family="CMU Serif", las=1, bty="l", cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 5))

# Note, the below code is modified from the implementation at:
# https://github.com/adChong/bc-stan/blob/master/src/main.R
# The experimental data and discrepancy have been stripped out of this to leave only an emulator.
# In addition, a much more efficient code for making predictions with the emulator has been implemented.

# get dimensions of the various datasets

# Note. I have continued to make the distinction between "controlled" inputs, x,
# and "uncontrolled" inputs, t, in this implementation. This isn't strictly necessary for the emulator,
# which just sees inputs, and does not care what type of inputs these are. I have kept this 
# distinction to minimise the number of changes to the formatting of inputs when switching between
# emulator and calibration codes.
p = 1                   # number of "controlled" inputs, x
q = 1                   # number of "uncontrolled" inputs, t
m = nrow(XT_sim)        # sample size of computer simulation data

# extract data from various tables
eta = dt_simulation$D           # simulation output (emulator training data)
# separate "controlled" and "uncontrolled" inputs in emulator training data
xc = as.matrix(XT_sim$x_sim)    # "controlled" input
tc = as.matrix(XT_sim$d)        # "uncontrolled" input

# separate "controlled" and "uncontrolled" inputs at which predictions are required
x_pred = as.matrix(xt$x)  # "controlled" input
t_pred = as.matrix(xt$t)  # "uncontrolled" input
n_pred = nrow(x_pred)   # store required number of predictions

# standardise ouputs to have zero sample mean unit sample variance. 
eta_mu = mean(as.matrix(eta), na.rm = TRUE) # mean value
eta_sd = sd(as.matrix(eta), na.rm = TRUE)   # standard deviation
eta = (eta - eta_mu)/eta_sd

# Standardise controlled and uncontrolled inputs to lie on [0,1]
# start with controlled inputs x
x_min = as.matrix(colMins(xc))
x_max = as.matrix(colMaxs(xc))
xc = (xc - matrix(rep(x_min,m,ncol=p,byrow=TRUE)))/matrix(rep(x_max-x_min,m,ncol=p,byrow=TRUE))
x_pred = (x_pred - matrix(rep(x_min,n_pred,ncol=p,byrow=TRUE)))/matrix(rep(x_max-x_min,n_pred,ncol=p,byrow=TRUE))
# now also standardise calibration inputs
t_min = as.matrix(colMins(tc))
t_max = as.matrix(colMaxs(tc))
tc = (tc - matrix(rep(t_min,m,ncol=q,byrow=TRUE)))/matrix(rep(t_max-t_min,m,ncol=q,byrow=TRUE))
t_pred = (t_pred - matrix(rep(t_min,n_pred,ncol=q,byrow=TRUE)))/matrix(rep(t_max-t_min,n_pred,ncol=q,byrow=TRUE))

# create data as list for input to Stan
stan_data = list(m=m, n_pred=n_pred, p=p, q=q, eta=c(as.matrix(eta)), xc=as.matrix(xc), 
                 x_pred=as.matrix(x_pred), tc=as.matrix(tc), t_pred=as.matrix(t_pred))

# run code for fitting Bayesian emulator, and making subsequent predictions
fit0 = stan(file = "C:/Users/cs2361/Documents/R/Gaussian Processes/Bayesian_Emulator.stan",
            data = stan_data,
            iter = 4000,
            chains = 3)

#---------------------------------------------------------------------------------------------------------------------

# This section of code for post-processing, and generation of plots of predictions

# plot traceplots, excluding warm-up
dev.new(noRStudioGD = TRUE)  # generate plots in separate window
stan_trace(fit0, pars = c("rho_eta", "lambda_eta"))

# summarise results to check convergence
print(fit0, pars = c("beta_eta", "lambda_eta","f_pred"))

# Extract data from STAN model
samples <- extract(fit0)
N_samples = dim(samples$rho_eta)[1] # get total number of samples

# extract predicted displacements and transform these onto their correct scale
# (recalling that observations were transformed to have zero sample mean, and 
# unit sample variance)
f_pred = samples$f_pred
f_trans = f_pred*eta_sd + eta_mu

# Produce plots of posterior and prior distributions of correlation lengths,
# (rho), and precision (lambda)
dev.new(noRStudioGD = TRUE)
par(mfrow = c(1,3))
axis_labels = list("rho_eta1","rho_eta2")
titles = list("correlation length x coordinate","correlation length depth")
rho_plot = seq(0,1, length.out = 100)
for (i in 1:2){
  # Plot histogram of posterior distribution
  hist(samples$rho_eta[,i],
       main = titles[i],
       xlab = axis_labels[i],
       col = "firebrick1",
       breaks = 25,
       freq = FALSE,
       xlim = c(0,1), 
       cex.lab=1.5,
       cex.axis=1.5)
  # Plot prior distribution
  prior_plot = dbeta(rho_plot,shape1=1,shape2=0.3)
  lines(rho_plot,prior_plot,lwd=3,col="blue")
}

# marginal precision of emulator
# plot posterior
hist(samples$lambda_eta,
     main = "lambda_eta",
     xlab = "lambda_eta",
     col = "firebrick1",
     breaks = 25,
     freq = FALSE,
     xlim = c(0,2.5),
     cex.axis=1.5,
     cex.lab=1.5)
# plot prior
lambda_plot = seq(0,2.5, length.out = 100)
prior_plot = dgamma(lambda_plot,shape=10,rate=10)
lines(lambda_plot,prior_plot,lwd=3,col="blue")

# plot an individual sample prediction
# given that each sample is a 2D surface it would be too mess to plot lots of these,
# so I only plot 1. Other samples can be plotted through modifying "samInd"
samInd = 1; # index of which sample to plot (modify to look at samples)

# Create input and output grids for producing surface plots
x_surf <- matrix(xt$x, nrow = length(x_grid), ncol = length(t_grid)) # x coordinate
y_surf<- matrix(xt$t, nrow = length(x_grid), ncol = length(t_grid)) # beam depth
z_surf <- matrix(f_trans[samInd,], nrow = length(x_grid), ncol = length(t_grid)) # emulator sample

dev.new(noRStudioGD = TRUE) # generate plot in a new window
# use the 3D scatter function to generate a surface plot of the emulator sample.
# This function is useful as it makes it possible to overlay a 3D scatter plot of 
# the training data over the surface plot
scatter3D(dt_simulation$x_sim, dt_simulation$d, dt_simulation$D, main = "Individual Sample Realisation", xlab = "x (m))", xlim = c(0, L), ylab = "thickness deviation", ylim = c(-0.1*d,0.1*d), zlab = "displacement", zlim = c(min(z_surf),max(z_surf)), col = "black", add = FALSE, clab = "displacement",pch=18,bty="g",ticktype = "detailed",nticks = 6,surf = list(x = x_surf, y = y_surf, z = z_surf, facets = TRUE, col = ramp.col(c("yellow","red"))))
# also create an interactive object which may be rotated etc
plotrgl()

# Plot the mean prediction. This averages across all of our uncertainty. Note that this is
# (I think) equivalent to taking a sample from the closed-form expression for the emulator
# mean, and averaging across the posterior distributions for the hyperparameters
f_trans_mu = colMeans(f_trans)
# create output grid for surface plot
z_surf <- matrix(f_trans_mu, nrow = length(x_grid), ncol = length(t_grid))
dev.new(noRStudioGD = TRUE) # generate plot in a new window
scatter3D(dt_simulation$x_sim, (d+dt_simulation$d)*1000, dt_simulation$D, main = "Emulator Mean", xlab = "x (m)", xlim = c(0, L), ylab = "depth (mm)", ylim = c(900*d,1100*d), zlab = "displacement", zlim = c(min(z_surf),max(z_surf)), col = "black", add = FALSE, clab = "displacement",pch=18,bty="g",ticktype = "detailed",nticks = 6,surf = list(x = x_surf, y = 1000*(d+y_surf), z = z_surf, facets = TRUE, col = ramp.col(c("yellow","red"))),cex.lab=1.5,cex.axis=1.5)
plotrgl()

# plot standard deviations of samples. Note, I don't think this is equivalent to the emulator uncertainty
# (for this we would require the mean of the standard deviation at each point in the prediction grid, 
# which would require a more complex method of making predictions through the simulation in stan.)
# An alternative way of doing this is to plug the posterior distributions for the emulator hyperparameters
# into closed-form expressions for the predictive emulator covariance, then taking an average. I have code
# for doing this but have omitted it here for conciseness. This could also be outputted from the stan code
# with some minor modifications. Taking the standard deviation of the predictions still gives a quick, and
# decent point-wise measure of the emulator uncertainty. 
f_trans_sd = colSds(f_trans) # determine the standard deviation at each point
# create output grid for surface plot
z_surf <- matrix(f_trans_sd, nrow = length(x_grid), ncol = length(t_grid))
dev.new(noRStudioGD = TRUE) # generate plot in a new window
scatter3D(dt_simulation$x_sim, dt_simulation$d, rep(0,n_maximin), main = "Emulator Uncertainty", xlab = "x (m))", xlim = c(0, L), ylab = "thickness deviation", ylim = c(-0.1*d,0.1*d), zlab = "displacement", zlim = c(0,max(z_surf)), col = "black", add = FALSE, clab = "displacement",pch=18,bty="g",ticktype = "detailed",nticks = 6,surf = list(x = x_surf, y = y_surf, z = z_surf, facets = TRUE, col = ramp.col(c("yellow","red"))))
plotrgl()