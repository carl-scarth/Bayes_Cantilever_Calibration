import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from matplotlib import rcParams
import pymc as pm
import arviz as az
import sys

# Add the src directory to the pythonpath for loading shared modules
src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from sample_prior import *

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
    x_train = transformed_LHS(inputs, N_train, sampler_package="scikit-optimize", sampler_kwargs={"lhs_type":"classic","criterion":"maximin", "iterations":10000})
    inp_str = [inp[0] for inp in inputs] # List of input variable names

    # Plotting parameters
    rcParams.update({'figure.figsize' : (8,6),
                    'font.size': 16,
                    'figure.titlesize' : 18,
                    'axes.labelsize': 18,
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 15})    

# ------------------------------------------------------------------------------
#           Run the cantilever beam model for Design of experiments
# ------------------------------------------------------------------------------
    y_train = np.empty(shape=N_train)
    for i, x_i in enumerate(x_train):
        # x is taken as L to output tip deflection
        y_train[i] = cantilever_beam(x=x_i[4], E=x_i[0], b=x_i[2], d=x_i[3], P=x_i[1], L=x_i[4])

# ------------------------------------------------------------------------------
#          Generate a set of points at which predictions are required
# ------------------------------------------------------------------------------

    N_pred = 2000  # Number of predictions
    x_pred = sample_prior(inputs, N_pred)  # Generate a set of random samples
    # Run the cantilever model for the prediction points (for validation)
    y_pred = np.empty(shape=N_pred)
    for i, x_i in enumerate(x_pred):
        # x is taken as L to output tip deflection
        y_pred[i] = cantilever_beam(x=x_i[4], E=x_i[0], b=x_i[2], d=x_i[3], P=x_i[1], L=x_i[4])

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
    sns.pairplot(plot_frame, vars = inp_str, hue="Category", palette = sns.color_palette("viridis",N_grades), diag_kind=None, plot_kws=dict(s = 50))
    # Produce a pairs plot for the points at which predictions are required
    sns.pairplot(pd.DataFrame(x_pred, columns=inp_str))

# ------------------------------------------------------------------------------
#                           Standardise the data
# ------------------------------------------------------------------------------

    # Standardise outputs to have zero mean and unit sample variance
    #y_mu = np.mean(y_train)
    #y_sd = np.std(y_train)
    #y_trans = (y_train - y_mu)/y_sd
    
    # Alternative code using scikit-learn
    y_scaler = StandardScaler()
    y_trans = y_scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1)
    print(y_trans)

    # Normalise inputs such that training data lies on the unit hypercube
    # x_min = x_train.min(axis = 0)
    # x_max = x_train.max(axis = 0)
    # x_trans = (x_train - x_min)/(x_max- x_min)
    # Normalise test data in the same fashion as the training data for consistency
    # x_pred_trans = (x_pred - x_min)/(x_max- x_min)
    
    # Alternative code using scikit-learn
    x_scaler = MinMaxScaler()
    x_trans = x_scaler.fit_transform(x_train)
    x_pred_trans = x_scaler.transform(x_pred)
    
#-------------------------------------------------------------------------------
#              Fit emulator using Maximum Likelihood Estimation
#-------------------------------------------------------------------------------

    kernel = ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-5,10.0))*RBF(length_scale=tuple(0.5 for i in range(d)), length_scale_bounds=(1e-5,10.0))
    gp_MLE = GaussianProcessRegressor(kernel=kernel, alpha = 1e-8, normalize_y = False, n_restarts_optimizer=1000).fit(x_trans, y_trans)

    print(gp_MLE)
    sasdsads
    # Happy with the Kernel - consider reducing the number of restarts in the regressor
    # Make some predictions and expose the optimised hyperparameters

    # Keep looking at Kernel functions - check if the paameters I've used make sense, also make predictions. Could also try using default values
    # Note, I might have broken the below code by changing to scikit learn preprocessing which assumes different shape to pymc

#-------------------------------------------------------------------------------
#                    Fit emulator using Bayesian inference 
#-------------------------------------------------------------------------------

    # Create pymc model
    with pm.Model() as emulator_model:
    
        # Priors on emulator hyperparameters and noise parameter
        # Updata to improve convergence. Stan guidance on GP priors:
        #'https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations#priors-for-gaussian-processes'
        # Example I'm following:
        # https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html
              
        # Scale parameter
        # lambda_em = pm.Gamma("lambda_em", alpha = 5.0, beta = 5.0)
        sigma_em = pm.HalfNormal("sigma_em", sigma = 1.0)
                
        # Define correltion length parameters
        # rho = pm.Beta("rho", alpha = 1.0, beta = 0.1, shape = 3)
        # beta = pm.Deterministic("beta", -4.0*np.log(rho))
        ls = pm.InverseGamma("ls", alpha = 4.0, beta = 4.0, shape = d)
        # Define mean and covariance functions
        mean_func = pm.gp.mean.Zero()
        # cov_func = pm.gp.cov.Constant(1.0/lambda_em) * pm.gp.cov.ExpQuad(3,ls=2.0/pt.sqrt(beta)) # + pm.gp.cov.WhiteNoise(sigma_noise**2)
        cov_func = pm.gp.cov.Constant(sigma_em**2) * pm.gp.cov.ExpQuad(d,ls=ls)
        # In the long run I'd rather be able to implement my own covariance matrices,
        # but this requires figuring out pytensor. Look through documentation.
        # Also discussion topics might help
    
        #----------------------------------------------------------------------
        
        # Implementation using multivariate normal:
        # y_ = pm.MvNormal("y", mu=mean_func(x_trans), cov=cov_func(x_trans), observed = y_trans)
            
        #----------------------------------------------------------------------
            
        # Alternative example using marginal likelihood GP
        #define marginal likelihood GP
        gp = pm.gp.Marginal(mean_func = mean_func, cov_func = cov_func)
        y_ = gp.marginal_likelihood("y", X = x_trans, y = y_trans, sigma = 1e-8)
    
        #--------------------------------------------------------------------------
    
        # I would like to explicitly model this, but don't know how to explicitly
        # define a pytensor object for the covariance matrix. I think I should
        # be able to do this once I have sussed out pytensor. Not sure that this
        # library is actually that good for readability
        # Atttempt at implementing this so far...
        #Sigma_em = np.zeros((N,N))
        #for i in range(N):
            #    for j in range(N):
                #        dx = x_trans[i,] - x_trans[j,]
                #        print(type(beta))
                #        print(beta.shape)
                #        print(beta*dx*dx)
                #        print((beta*dx*dx).shape)
                #        print(sum(beta*dx*dx).shape)
                #        test = sum(beta*dx*dx)
                #        Sigma_em[i, j] = sum(beta * dx * dx)
                #        Sigma_em[j, i] = Sigma_em[i, j]
            
                # We can implement a Gaussian process using the Latent (without)
                # noise or marginal (with noise) Gaussian process modules in 
                # pymc. It's probably also possible using a multivariate normal,
                # but I'd need to understand the tensor data structure a little 
                # better to do this
 
    #    Sigma_em = np.exp(-Sigma_em[i, j])/lambda_em + np.identity(N)*sigma_noise
    #    

        # Draw 3000 posterior samples using NUTS sampling
        # default target_accept = 0.9. Increase if needed to improve convergece at
        # the cost of higher sampling time
        idata = pm.sample(3000, target_accept=0.9)
    
    az.plot_trace(idata, combined=True, figsize=(10, 7))
    
#------------------------------------------------------------------------------
#                 Plot histograms of the posterior marginals
#------------------------------------------------------------------------------

    # Extract samples of various quantities 
    # rho = idata.posterior["rho"].values.reshape((-1,3))    
    # beta= idata.posterior["beta"].values.reshape((-1,3))
    ls = idata.posterior["ls"].values.reshape((-1,d))
    # lambda_em = idata.posterior["lambda_em"].values.reshape((-1,1))
    sigma_em = idata.posterior["sigma_em"].values.reshape((-1,1))

    fig, ax = plt.subplots(figsize=[16,6])
    #ax.hist(lambda_em, bins = 49, color="tab:green",edgecolor="black")
    #ax.set_title("Emulator Precision")
    
    ax.hist(sigma_em, bins = 49, density = True, color="tab:green",edgecolor="black")
    ax.set_title("Emulator standard deviation")
    ax.set_xlabel("Sigma_em")
    ax.set_ylabel("Density")

    fig2, axs2 = plt.subplots(1,d,figsize=[20,6])
    for i, ax2 in enumerate(axs2):
    #    ax2.hist(rho[:,i], bins = 49, color = "tab:blue", edgecolor="black")
        ax2.hist(ls[:,i], bins = 49, color = "tab:blue", edgecolor="black")
        ax2.set_xlabel("ls_" + str(i))
        ax2.set_ylabel("Density")
    #fig3, axs3 = plt.subplots(1,3, figsize=[16,6])
    # for i, ax3 in enumerate(axs3):
    #ax3.hist(beta[:,i], bins = 49, color = "tab:orange", edgecolor="black")
  
    N_post = sigma_em.shape[0] # Number of posterior samples
    
    # Better to use gp.predict, as unlike the conditional method, this may be 
    # used outside of the "with" statement
    # This only generates the means and covariance, though this could in turn be
    # used to sample

#------------------------------------------------------------------------------
#                 Make predictions using the fitted emulator
#------------------------------------------------------------------------------

    # Take a sub-sample of the posterior parameters (Note: If the number of 
    # predictions is greater than the numver of posterior samples it might be
    # necessary to set replace = True)
    N_sub_sam = 50
    pred_ind = np.random.choice(N_post,N_sub_sam, replace = False)
    mu_pred = np.empty((N_pred, N_sub_sam),float)
    sigma_pred = np.empty((N_pred, N_sub_sam),float)
    # f_pred = np.empty((N_pred, N_sam_pred),float)

    # Take N_sam_pred random samples from the posterior and generate a sample prediction
    for i, post_i in enumerate(pred_ind):
        point = {"sigma_em" : float(sigma_em[post_i]), "ls" : ls[post_i,]}
        # Note, I could just used the closed form expression for this and it would work without the irritating pymc formatting
        # Could also use scikit-learn
        mu_pred[:,i], sigma_pred[:,i] = gp.predict(x_pred_trans, point = point, model = emulator_model, diag = True)
        # For if the number of predictions is low enough that it's possible to calculate the whole predictive covariance matrix
        # mu_pred[:,i], cov = gp.predict(x_pred_trans, point = point, model = emulator_model, diag = False)
        # f_pred[:,i] = np.random.multivariate_normal(mu_pred[:,i], cov)
        # sigma_pred[:,i] = np.diagonal(cov)
        
    mu_pred = np.mean(mu_pred,axis=1)
    sigma_pred = np.mean(sigma_pred,axis=1)

#------------------------------------------------------------------------------
#           Transform back onto correct scale then plot histogram
#------------------------------------------------------------------------------

    # Take average across posterior sample

# Compare against MLE? Add point-estimate to plot?
# Compare histogram
    # Convert the mean and standard deviation back onto the true scale
    # mu_pred = mu_pred*y_sd + y_mu
    # sigma_pred = sigma_pred*y_sd
    mu_pred = y_scaler.inverse_transform(mu_pred.reshape(-1,1)).reshape(-1)
    print(mu_pred)
    # with_mean might not work in this context
    sigma_pred = y_scaler.inverse_transform(sigma_pred.reshape(-1,1), with_mean = False).reshape(-1)
    print(sigma_pred)

    fig3, ax3 = plt.subplots(figsize=[10,6])
    # Plot histogram of true displacement
    ax3.hist(y_pred, bins = 49, color = "salmon", edgecolor="black", density=True, label = "Beam model")
    # Slightly reduce the smoothness of the kde, and truncate a little belong extreme values
    sns.kdeplot(mu_pred, color="blue", linewidth=3, bw_adjust=0.95, cut=0.1, ax=ax3, label = "Emulator")
    ax3.set_xlabel("Tip displacement (mm)")
    ax3.set_ylabel("Density")
    ax3.set_title("Uncertainty propagation results")
    ax3.legend()

    fig4,ax4 = plt.subplots(figsize=[10,6])
    ax4.hist(sigma_pred, bins=49, color = "tab:red", edgecolor="black", density=True)
    ax4.set_xlabel("Emulator standard deviation (m)")
    ax4.set_ylabel("Density")
    ax4.set_title("Emulator standard deviation")
    
    plt.show()

    # Put point estimate against posteriors
    # Play aroud with other formulations of model
    # Also add prior plots on posterior histograms.
    # Ideally I would define the prior outside of the context menu then sample some draws from this so autmoated. (see marginal_likelihood example in Thomas' folder)
    # It isn't possible to do this outside of the context menu
    # Just manually implement for now
    # Consider working with pandas
    # This code is pretty good. Move onto other example then tidy up later