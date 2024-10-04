# Fits a Gaussian process emulator to the point-wise displacement of a simple cantilever beam
# with uncertain depth

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pymc as pm
import arviz as az
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from scipy.stats import halfnorm, invgamma

src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from maximin import *
from transform_input_output import normalise_inputs, standardise_output, rescale_output
import utils

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # Define space of inputs, generate training samples, and set up model parameters
    # ------------------------------------------------------------------------------

    E = 68E9 # Young's modulus
    P = 10.0 # Applied load
    b = 0.04 # Beam width
    L = 1.0  # Beam length
    d = 0.01 # Nominal value of  beam height d

    N_train = 25  # Number of required training data points
    
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
    
    # Set plotting parameters
    utils.set_plot_params()

    # ------------------------------------------------------------------------------
    #           Run the cantilever beam model for Design of experiments
    # ------------------------------------------------------------------------------
    
    # Define a list of coordinates at which the simulation output is generated.
    sim_coords = [i/(N_plot-1)*L for i in range(N_plot)]
    
    # Run the cantilever model n_simulations times
    y_all = np.empty(shape = (N_plot,N_train))
    for i, x_i in enumerate(x_train):
        y_all[:,i] = cantilever_beam(x=sim_coords,E=E,b=b,d = x_i[0],P=P, L=L)

    # Plot the output
    fig, ax = plt.subplots()
    ax.plot(sim_coords,y_all)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title("Beam displacement across training samples")

    # Reshape all n_simulations x N_train points into a vector
    y_all = y_all.T.reshape((-1))
    
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
#                        Plot the Design of Experiments
# ------------------------------------------------------------------------------
    
    N_grades = 8 # Number of different values to divide the output into
    
    # Create data frame as Seaborn only works with Pandas
    #plot_frame = pd.concat((pd.DataFrame(x_train,columns=inp_str),pd.DataFrame(y_train,columns=["Displacement"])),axis=1)
    plot_frame = pd.concat((x_train, y_train),axis=1)
    # Bin the displacement into intervals, and create a new categorical column
    # in the dataframe stating which interval each point lies within
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
    
# ------------------------------------------------------------------------------
#          Generate a set of points at which predictions are required
# ------------------------------------------------------------------------------

    # Generate a sequence of points across x and t
    n_grid = 15
    x_grid = np.linspace(min(sim_coords),max(sim_coords), num = n_grid).reshape(-1)
    t_grid = np.linspace(0.9*d,1.1*d, num = n_grid)

    # Take the union of these points with the training data - we want to verify
    # that uncertainty is zero at these points
    #x_grid = np.sort(np.unique(np.concatenate((x_grid,x_train[:,1]))))
    #t_grid = np.sort(np.unique(np.concatenate((t_grid,x_train[:,0]))))
    x_grid = np.sort(np.unique(np.concatenate((x_grid,x_train["x"].to_numpy()))))
    t_grid = np.sort(np.unique(np.concatenate((t_grid,x_train["d"].to_numpy()))))
    # Expand points into a grid
    x_grid, t_grid = np.meshgrid(x_grid,t_grid)
    
    #x_pred = np.concatenate((t_grid.reshape(-1,1),x_grid.reshape(-1,1)),axis=1)
    x_pred = pd.DataFrame({"d" : t_grid.reshape(-1), "x" : x_grid.reshape(-1)})
    N_pred = x_pred.shape[0]

# ------------------------------------------------------------------------------
#                           Standardise the data
# ------------------------------------------------------------------------------

    # Standardise outputs to have zero mean and unit sample variance
    y_trans, y_mu, y_sd = standardise_output(y_train)

    # Normalise inputs such that training data is on the unit hypercube
    x_trans, x_min, x_max = normalise_inputs(x_train)

    # Normalise test data in the same way as the training data for consistency
    x_pred_trans = normalise_inputs(x_pred, x_min = x_min, x_max = x_max)
    
#-------------------------------------------------------------------------------
#              Fit emulator using Maximum Likelihood Estimation
#-------------------------------------------------------------------------------

    # Construct a Kernel
    kernel = ConstantKernel(constant_value=1.0,constant_value_bounds=(1e-5,100.0))*RBF(length_scale=tuple(0.5 for i in range(2)), length_scale_bounds=(1e-5,10.00))
    # Create a Gaussian Process regressor object and fit to training data
    gp_MLE = GaussianProcessRegressor(kernel=kernel, alpha = 1e-8, normalize_y = False, n_restarts_optimizer = 50).fit(x_trans, y_trans)
    # Return predictive mean and standard deviation
    mu_pred_MLE, sigma_pred_MLE = gp_MLE.predict(x_pred_trans, return_std=True)
    # Rescale predictions
    mu_pred_MLE = rescale_output(mu_pred_MLE, mu_y = y_mu, sigma_y = y_sd)
    sigma_pred_MLE = rescale_output(sigma_pred_MLE, sigma_y=y_sd, std=True)
    # Extract hyperparameters of fitted Gaussian process
    sigma_em_MLE = np.sqrt(gp_MLE.kernel_.get_params()["k1__constant_value"])
    # Is there a factor of 2 in one of the covariance functions?
    ls_MLE = gp_MLE.kernel_.get_params()["k2__length_scale"]

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
                
        # Define correlation length parameters
        # rho = pm.Beta("rho", alpha = 1.0, beta = 0.1, shape = 3)
        # beta = pm.Deterministic("beta", -4.0*np.log(rho))
        ls = pm.InverseGamma("ls", alpha = 4.0, beta = 4.0, shape = 2)
        # Define mean and covariance functions
        mean_func = pm.gp.mean.Zero()
        # cov_func = pm.gp.cov.Constant(1.0/lambda_em) * pm.gp.cov.ExpQuad(3,ls=2.0/pt.sqrt(beta)) # + pm.gp.cov.WhiteNoise(sigma_noise**2)
        cov_func = pm.gp.cov.Constant(sigma_em**2) * pm.gp.cov.ExpQuad(2,ls=ls)

        # In the long run I'd rather be able to implement my own covariance matrices,
        # but this requires figuring out pytensor. Look through documentation.
        # Also discussion topics might help
    
        #----------------------------------------------------------------------
        
        # Implementation using multivariate normal:
        # y_ = pm.MvNormal("y", mu=mean_func(x_trans).eval(), cov=cov_func(x_trans).eval(), observed = y_trans)
            
        #----------------------------------------------------------------------
            
        # Alternative example using marginal likelihood GP
        #define marginal likelihood GP
        gp = pm.gp.Marginal(mean_func = mean_func, cov_func = cov_func)
        #y_ = gp.marginal_likelihood("y", X = x_trans, y = y_trans, sigma = 1e-8)
        y_ = gp.marginal_likelihood("y", X = x_trans.values, y = y_trans.values, sigma = 1e-8)
    
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
    
    az.plot_trace(idata, combined=True, figsize=(10, 8))

#------------------------------------------------------------------------------
#                 Plot histograms of the posterior marginals
#------------------------------------------------------------------------------

    # Extract samples of various quantities 
    # rho = idata.posterior["rho"].values.reshape((-1,3))    
    # beta= idata.posterior["beta"].values.reshape((-1,3))
    
    ls = idata.posterior["ls"].values.reshape((-1,2))
    # lambda_em = idata.posterior["lambda_em"].values.reshape((-1,1))
    sigma_em = idata.posterior["sigma_em"].values.reshape((-1,1))

    fig3, ax3 = plt.subplots()
    #ax.hist(lambda_em, bins = 49, color="tab:green",edgecolor="black")
    #ax.set_title("Emulator Precision")
    
    sigma_plot = np.linspace(0.0,10.0,200)
    ax3.hist(sigma_em, bins = 49, density = True, color="tomato",edgecolor="black")
    ax3.plot(sigma_plot, halfnorm.pdf(sigma_plot), lw = 2, color = "blue") # Add prior plot
    ax3.plot([sigma_em_MLE, sigma_em_MLE], [0, ax3.get_ylim()[1]], "--k", linewidth=2)
    ax3.annotate("MLE solution", xy = (sigma_em_MLE, 0.5*ax3.get_ylim()[1]), xytext = (sigma_em_MLE-2.0, 0.75*ax3.get_ylim()[1]), arrowprops = dict(facecolor="black",shrink=0.05, width = 1.0))
    ax3.set_title("Emulator standard deviation")
    ax3.set_xlabel("Sigma_em")
    ax3.set_ylabel("Density")


    fig4, axs4 = plt.subplots(1,2,figsize=[10,6])
    ls_plot = np.linspace(0.0,4.0,100)
    for i, ax4 in enumerate(axs4):
        ax4.hist(ls[:,i], bins = 49, density = True, color = "tomato", edgecolor="black")
        ax4.plot([ls_MLE[i], ls_MLE[i]],  [0, 0.9*ax4.get_ylim()[1]], "--k", linewidth=2)
        ax4.annotate("MLE solution", xy = (ls_MLE[i], 0.5*ax4.get_ylim()[1]), xytext = (0.1*ax4.get_xlim()[1], 0.95*ax4.get_ylim()[1]), arrowprops = dict(facecolor="black",shrink=0.05, width = 1.0))
        # Note: I think the scipy documentation is incorrect about the inverse gamma pdf
        # I've compared with my own implementation of that described in the pymc docs and
        # it matches, so I'm confident the below correctly matches the prior
        ax4.plot(ls_plot, invgamma.pdf(ls_plot, 4.0, loc = 0.0, scale = 4.0), lw = 2, color = "blue")
        ax4.set_xlabel("ls_" + str(i))
        ax4.set_ylabel("Density")
  
    N_post = sigma_em.shape[0] # Number of posterior samples
    
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
    # f_pred = np.empty((N_pred, N_sub_sam),float)

    # Take N_sam_pred random samples from the posterior and generate a sample prediction
    for i, post_i in enumerate(pred_ind):
        point = {"sigma_em" : float(sigma_em[post_i]), "ls" : ls[post_i,]}
        # Note, I could just used the closed form expression for this and it would work without the irritating pymc formatting
        # Could also use scikit-learn
        mu_pred[:,i], sigma_pred[:,i] = gp.predict(x_pred_trans.values, point = point, model = emulator_model, diag = True)
        # For if the number of predictions is low enough that it's possible to calculate the whole predictive covariance matrix
        # mu_pred[:,i], cov = gp.predict(x_pred_trans, point = point, model = emulator_model, diag = False)
        # f_pred[:,i] = np.random.multivariate_normal(mu_pred[:,i], cov)
        # sigma_pred[:,i] = np.diagonal(cov)
    
    sigma_pred = np.sqrt(sigma_pred)
    mu_pred = np.mean(mu_pred,axis=1)
    sigma_pred = np.mean(sigma_pred,axis=1)

#------------------------------------------------------------------------------
#           Transform back onto correct scale then plot predictions
#------------------------------------------------------------------------------

    # Take average across posterior sample

    # Convert the mean and standard deviation back onto the true scale
    mu_pred = rescale_output(mu_pred, mu_y = y_mu, sigma_y = y_sd)
    sigma_pred = rescale_output(sigma_pred, sigma_y = y_sd, std = True)

    fig5, ax5 = plt.subplots(1,2,subplot_kw = {"projection" : "3d"})
    ax5[0].plot_surface(x_grid, t_grid, mu_pred.reshape(x_grid.shape), rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha = 0.8)
    # ax5.scatter(x_train[:,1], x_train[:,0], y_train, s = 25, color = "red", marker = "x")
    ax5[0].scatter(x_train["x"], x_train["d"], y_train, s = 25, color = "red", marker = "x")
    ax5[0].set_xlabel("x (m)", labelpad = 14.0)
    ax5[0].set_ylabel("Beam height (m)", labelpad = 14.0)
    ax5[0].set_zlabel("Displacement (m)", labelpad = 14.0)
    ax5[0].set_title("Bayesian Emulator Mean")
    ax5[1].plot_surface(x_grid, t_grid, mu_pred_MLE.reshape(x_grid.shape), rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False, alpha = 0.8)
    ax5[1].scatter(x_train["x"], x_train["d"], y_train, s = 25, color = "red", marker = "x")
    ax5[1].set_xlabel("x (m)", labelpad = 14.0)
    ax5[1].set_ylabel("Beam height (m)", labelpad = 14.0)
    ax5[1].set_zlabel("Displacement (m)", labelpad = 14.0)
    ax5[1].set_title("MLE Emulator Mean")

    fig6, ax6 = plt.subplots(1,2,subplot_kw = {"projection": "3d"})
    ax6[0].plot_surface(x_grid, t_grid, sigma_pred.reshape(x_grid.shape), rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    # ax6.scatter(x_train[:,1], x_train[:,0], np.zeros(x_train.shape[0]), s = 25, color = "red", marker = "x")
    ax6[0].scatter(x_train["x"], x_train["d"], np.zeros(x_train.shape[0]), s = 25, color = "red", marker = "x")
    ax6[0].set_xlabel("x (m)", labelpad = 14.0)
    ax6[0].set_ylabel("Beam depth (m)", labelpad = 14.0)
    ax6[0].set_zlabel("Standard deviation (m)", labelpad = 14.0)
    ax6[0].set_title("Bayesian Emulator Standard deviation")
    ax6[0].ticklabel_format(useOffset=False, style = "plain") # Keep in standard format, not scientific
    ax6[1].plot_surface(x_grid, t_grid, sigma_pred_MLE.reshape(x_grid.shape), rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    # ax6.scatter(x_train[:,1], x_train[:,0], np.zeros(x_train.shape[0]), s = 25, color = "red", marker = "x")
    ax6[1].scatter(x_train["x"], x_train["d"], np.zeros(x_train.shape[0]), s = 25, color = "red", marker = "x")
    ax6[1].set_xlabel("x (m)", labelpad = 14.0)
    ax6[1].set_ylabel("Beam depth (m)", labelpad = 14.0)
    ax6[1].set_zlabel("Standard deviation (m)", labelpad = 14.0)
    ax6[1].set_title("MLE Emulator Standard deviation")
    ax6[1].ticklabel_format(useOffset=False, style = "plain") # Keep in standard format, not scientific
    # Non-zero standard deviaton close to jitter (scaled like the rest. would need to implement own
    # expression to avoid this)

    # Put point estimate against posteriors
    # Play aroud with other formulations of model
    # Ideally I would define the prior outside of the context menu then sample some draws from this so autmoated. (see marginal_likelihood example in Thomas' folder)
    plt.tight_layout()
    plt.show()

    #-------------------------------------------------------------------------------

# what about numpyro?
# Experiment to see if pymc any quick than stan for large N_maximin
