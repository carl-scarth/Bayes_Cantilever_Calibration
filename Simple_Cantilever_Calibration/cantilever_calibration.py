# Calibration of a simple cantilver beam model with uncertain depth using synthetic data
import sys
import numpy as np
import pandas as pd
from scipy.stats import uniform, norm, halfnorm, invgamma, expon
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pymc as pm
import pytensor
import pytensor.tensor as tt
import arviz as az

src_path = "../source"
sys.path.insert(0, src_path)

from cantilever_beam import *  # Import cantilever model
from LHS_Design import transformed_LHS  # Import Latin Hypercube module
from maximin import *
from Prior import *
from transform_input_output import normalise_inputs, standardise_output, rescale_output, rescale_inputs
import utils

# Create a custom covariance function for Bayesian model calibration
class KennedyOHagan(pm.gp.cov.Covariance):
    def __init__(self, input_dim: int,
                 m: int,
                 n: int,
                 ls: tt.tensor,
                 sigma_eta: tt.tensor,
                 sigma_e: tt.tensor):
        
        super().__init__(input_dim) # Inherit characteristics of the pymc covariance class (declared in class definition), with input_dim given in integer
        self.m = m
        self.n = n
        self.ls = ls
        self.sigma_eta = sigma_eta
        self.sigma_e = sigma_e
        # Note: X and Xs aren't assigned until calling a method

    def diag(self):
        # Diagonal terms for the experimental data
        diag_data = tt.alloc(tt.square(self.sigma_eta) + tt.square(self.sigma_e) + 1e-4, self.n) # 1e-4 is square of the nugget
        # Now for just the emulator
        diag_em = tt.alloc(tt.square(self.sigma_eta) + 1e-4, self.m)
        return(tt.concatenate([diag_data, diag_em],axis=0))
    
    def full(self, X, Xs = None):
        # Xs is passed as a second input to the covariance
        # Return the cross-covariance between X and Xs
        cov_em = pm.gp.cov.Constant(self.sigma_eta**2) * pm.gp.cov.ExpQuad(2,ls=self.ls) + pm.gp.cov.WhiteNoise(1e-8)
        # ls does note stand for l^2, so actually covariance is exp(-(X-X')/(2*LS^2))
        if Xs is None:
            # This should be the main usage - should never have to calculate a cross covariance, unless maybe during prediction?
            # Leave else blank for now and see if it compiles...
            # Ideally I'd implement everything for clarity, but easier to use pymc functions as tensor operations are already implemented
            cov_full = cov_em(X)
            cov_e = pm.gp.cov.WhiteNoise(self.sigma_e)
            # WhiteNoise kernel automatically squares sigma
            error_diag = cov_e(X[0:self.n,:])
            # Can't use indexing with tensors, instead use subtensor.set_subtensor
            cov_full = tt.subtensor.set_subtensor(cov_full[0:self.n,0:self.n],cov_full[0:self.n,0:self.n]+error_diag)
        else:
            # Assumes Xs is to be used for prediction, and so there's no need to account for the observation error
            # Observation error only applies to the diagonal entries.
            Xs = tt.as_tensor_variable(Xs)
            cov_full = cov_em(X,Xs)
        
        return(cov_full)

if __name__ == "__main__":


    # ------------------------------------------------------------------------------
    # Define space of inputs, generate training samples, and set up model parameters
    # ------------------------------------------------------------------------------

    N_data = 10         # number of experimental data points
    N_repeats = 3       # number of repeated experimental data points
    N_train = 30        # number of emulator training data points
    N_plot = 100        # Number of points at which beam deflection is plotted
    N_maximin = 50      # Number of model output data points which are to be retained for training the emulator
    p = 1               # Number of controlled inputs
    q = 1               # Number of uncertain calibration inputs
  
    # Define magnitude of the synthetic noise which will be added to model outputs to get "observed" data
    sigma_noise = 0.00025 # standard deviation

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
    print("true d = "  + str(d_data))
  
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
    dt_data.delta = dt_data.delta + norm.rvs(loc=0, scale = sigma_noise, size = N_data+N_repeats)
    y = dt_data.delta
    n = y.shape[0]
    
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
    eta_all_plot = np.empty(shape = (N_plot, N_train))
    for i, x_i in enumerate(x_train):
        eta_all_plot[:,i] = cantilever_beam(x=sim_coords,E=E,b=b,d = x_i[0],P=P, L=L)
  
    # Reshape all n_simulations x N_train points into a vector
    eta_all = eta_all_plot.T.reshape((-1))
    
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
    eta_train = eta_all[ind]
  
    # If using Pandas dataframe (haven't changed any of the above code so may be able to do this also)
    x_train = pd.DataFrame(x_train, columns = inp_str)
    eta_train = pd.Series(eta_train, name="Displacement")
    m = x_train.shape[0] 

    # ------------------------------------------------------------------------------
    #              Plot the Design of Experiments and experimental data
    #-------------------------------------------------------------------------------
    
    # First plot beam displacement 
    # Plot observed displacements
    fig, ax = plt.subplots()
      
    # define points at which the "true" displacement is to be plotted
    x_plot = [i/(N_plot-1)*L for i in range(N_plot)]
    x_pred = np.array(x_plot, ndmin=2).T # We want to make calibrated predictions at the coordinates of the plot
    n_pred = x_pred.shape[0]   # number of predictions
    # run beam model to determine "true" displacement at these points
    delta_plot = cantilever_beam(x=x_plot,E=E,b=b,d=d_data,P=P,L=L)
    ax.plot(sim_coords,eta_all_plot[:,0],"-c", linewidth=0.5, label = "Prior simulation")
    ax.plot(sim_coords,eta_all_plot[:,1:],"-c", linewidth=0.5)
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
    plot_frame = pd.concat((x_train, eta_train),axis=1)
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    # Create a pairs plot of the training data, coloured according to the 
    # displacement value of each point
    fig2, axes2 = plt.subplots(1,2, sharey = True)
    fig2.suptitle("Training data before and after maximin search")
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes2[1], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))
    # There are only two inputs so it is more appropriate to use scatterplot thann pairplot
    plot_frame = pd.concat((pd.DataFrame(x_all, columns = inp_str),pd.DataFrame(eta_all,columns=["Displacement"])),axis=1)
    plot_frame["Category"] = pd.cut(plot_frame["Displacement"],N_grades)
    sns.scatterplot(x=inp_str[0], y=inp_str[1], data=plot_frame, ax=axes2[0], hue="Category", legend=False, palette = sns.color_palette("viridis",N_grades))

    # ------------------------------------------------------------------------------
    #                              Standardise the data
    # ------------------------------------------------------------------------------
    
    # Standardise outputs to have zero mean and unit sample variance
    eta_trans, mu_eta, sd_eta = standardise_output(eta_train)

    # Strandardise the experimental data using the same values for consistency
    y_trans = standardise_output(y, mu_y = mu_eta, sigma_y = sd_eta)

    # Normalise inputs such that training data is on the unit hypercube
    x_trans, x_min, x_max = normalise_inputs(x_train)

    # Normalise test data and field data in the same way as the training data for consistency
    xf_trans = normalise_inputs(dt_data.x.to_numpy().reshape(-1,1), x_min = x_min[1], x_max = x_max[1])
    x_pred_trans = normalise_inputs(x_pred, x_min = x_min[1], x_max = x_max[1])

#-------------------------------------------------------------------------------
#                 Perform calibration using Bayesian inference 
#-------------------------------------------------------------------------------

    # Put data in the correct format
    # Combine experimental data and model data
    y_eta = pd.concat((y_trans, eta_trans), axis = 0).to_numpy()
    # Total number of points
    N = m + n
    
    # Create pymc model
    # Shared tensors need to be created for all of the data which is shared across
    # different values of the uncertain inputs
    tf_shared = pytensor.shared(xf_trans)
    x_trans_shared = pytensor.shared(x_trans.to_numpy())
    with pm.Model() as calibration_model:
        # Useful examples I've followed to put together this model
        # Stan guidance on GP priors:
        #'https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations#priors-for-gaussian-processes'
        # Example I'm following:
        # https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Latent.html
        # Calibration examples:
        # https://discourse.pymc.io/t/bayesian-model-calibration-with-gaussian-process/948/19
        
        # Prior for field data
        theta = pm.Uniform("theta",lower=0.0,upper=1.0)
        # Create tensor variable for all controlled and uncontrolled experimental inputs
        theta_rep = tt.tile(theta,n, ndim=2).T
        XT_field = tt.concatenate([theta_rep, tf_shared], axis=1)
        # Also combine with model inputs
        XT = tt.concatenate([XT_field, x_trans_shared], axis=0)
        
        # Define hyperparameter priors
        sigma_eta = pm.HalfNormal("sigma_eta", sigma = 1.0) # emulator scale parameter
        sigma_e = pm.Exponential("sigma_e", lam = 10.0) # Observation error scale parameter
        ls = pm.InverseGamma("ls", alpha = 4.0, beta = 4.0, shape = 2) # correlation lengths
                
        # Define Gaussian process mean and covariance functions
        mean_func = pm.gp.mean.Zero()
        # Combine field data and experimental data into single matrix
        # cov_func_em = pm.gp.cov.Constant(sigma_eta**2) * pm.gp.cov.ExpQuad(2,ls=ls) + pm.gp.cov.WhiteNoise(1e-8)
        cov_func = KennedyOHagan(2, m, n, ls, sigma_eta, sigma_e)
        
        # Print output for debugging
        #print(f"theta = {theta.eval()}")
        #print("")
        #print(f"Combined model and experimental inputs = \n{XT.eval()}")
        #print("")
        #print(f"sigma_eta = {sigma_eta.eval()}")
        #print("")
        #print(f"sigma_e = {sigma_e.eval()}")
        #print("")
        #print(f"ls = {ls.eval()}")
        #print("")        
        #print(f"Mean vector = \n{mean_func(XT).eval()}")
        #print("")
        # print(mean_func.eval())
        #sigma_z = cov_func(XT)
        #print(f"Covariance matrix = \n{sigma_z.eval()}")

        # Example of how to debug with multiple samples
        #theta_test = pm.draw(theta, draws = 2)
        #print(theta_test)
        #print([theta_rep.eval({theta : theta_i}) for theta_i in theta_test])
    
        #----------------------------------------------------------------------
        
        # Implementation using multivariate normal: This seems less stable than the gp method, so I don't use it
        #y_ = pm.MvNormal("y", mu=mean_func(XT), cov=sigma_z, observed = y_eta)
        
        #----------------------------------------------------------------------
            
        # Implementation using marginal likelihood GP
        gp = pm.gp.Marginal(mean_func = mean_func, cov_func = cov_func)
        y_ = gp.marginal_likelihood("y", X = XT, y = y_eta, sigma = 1e-8)#
        
        #--------------------------------------------------------------------------
        # Draw 3000 posterior samples using NUTS sampling
        # default target_accept = 0.9. Increase if needed to improve convergece at
        # the cost of higher sampling time
        idata = pm.sample(3000, target_accept=0.9)
    
    # Produce MCMC trace plots
    az.plot_trace(idata, combined=True, figsize=(10, 8))

#------------------------------------------------------------------------------
#                 Plot histograms of the posterior marginals
#------------------------------------------------------------------------------

    # Extract output samples 
    ls_post = idata.posterior["ls"].values.reshape((-1,2))
    sigma_eta_post = idata.posterior["sigma_eta"].values.reshape((-1,1))
    sigma_e_post = idata.posterior["sigma_e"].values.reshape((-1,1))
    theta_post = idata.posterior["theta"].values.reshape((-1,1))


    # Plot standard deviations
    fig3, axes3 = plt.subplots(1,2)
    sigma_plot = np.linspace(0.0,5.0,200)
    axes3[0].hist(sigma_eta_post, bins = 49, density = True, color="tomato",edgecolor="black")
    axes3[0].plot(sigma_plot, halfnorm.pdf(sigma_plot), lw = 2, color = "blue") # Add prior plot
    axes3[0].set_title("Emulator standard deviation")
    axes3[0].set_xlabel("Sigma_eta")
    axes3[0].set_ylabel("Density")

    sigma_plot = np.linspace(0.0,0.5,200)
    axes3[1].hist(sigma_e_post, bins = 49, density = True, color="tomato",edgecolor="black")
    # Scipy implementaiton of exponential distribution has scale = 1/lambda
    axes3[1].plot(sigma_plot, expon.pdf(sigma_plot, scale = 0.1), lw = 2, color = "blue") # Add prior plot
    axes3[1].set_title("Observation error standard deviation")
    axes3[1].set_xlabel("Sigma_e")
    axes3[1].set_ylabel("Density")

    # Plot correlation lengths
    fig4, axs4 = plt.subplots(1,2,figsize=[10,6])
    ls_plot = np.linspace(0.0,4.0,100)
    for i, ax4 in enumerate(axs4):
        ax4.hist(ls_post[:,i], bins = 49, density = True, color = "tomato", edgecolor="black")
        # Note: I think the scipy documentation is incorrect about the inverse gamma pdf
        # I've compared with my own implementation of that described in the pymc docs and
        # it matches, so I'm confident the below correctly matches the prior
        ax4.plot(ls_plot, invgamma.pdf(ls_plot, 4.0, loc = 0.0, scale = 4.0), lw = 2, color = "blue")
        ax4.set_xlabel("ls_" + str(i))
        ax4.set_ylabel("Density")
        
    theta_trans = rescale_inputs(theta_post,x_min[0],x_max[0])
    fig5, ax5 = plt.subplots()
    theta_plot = np.linspace(0.9*d,1.1*d,100)
    ax5.hist(theta_trans, bins = 49, density = True, color = "tomato", edgecolor="black")
    ax5.plot(theta_plot, uniform.pdf((theta_plot - 0.9*d)/(0.2*d)), lw = 2, color = "blue")
    ax5.set_xlabel("theta")
    ax5.set_ylabel("Density")

    # Get number of posterior samples
    N_post = sigma_eta_post.shape[0] # Number of posterior samples

#------------------------------------------------------------------------------
#                   Make predictions using the fitted model
#------------------------------------------------------------------------------

    # Take a sub-sample of the posterior parameters (Note: If the number of 
    # predictions is greater than the number of posterior samples it might be
    # necessary to set replace = True)
    N_sub_sam = 100
    pred_ind = np.random.choice(N_post,N_sub_sam, replace = False)

    # Initialise output arrays
    mu_pred = np.empty((n_pred, N_sub_sam),float)
    sigma_pred = np.empty((n_pred, N_sub_sam),float)
    f_pred = np.empty((n_pred, N_sub_sam),float)

    with calibration_model:
        # Define separate covariance function for predictions without the observation error component
        cov_pred = pm.gp.cov.Constant(sigma_eta**2) * pm.gp.cov.ExpQuad(2,ls=ls)# + pm.gp.cov.WhiteNoise(1e-8)
        # Loop over posterior samples
        for i, post_i in enumerate(pred_ind):
            print("Calculating posterior prediction " + str(i))
            # Evaluate quadrant of covariance matrix associated only with training data (including observation error for experimental observations)
            x_train_i = XT.eval({theta: float(theta_post[post_i,:])})
            cov_train_i = cov_func(x_train_i).eval({sigma_eta : float(sigma_eta_post[post_i]), ls: ls_post[post_i,:], sigma_e : float(sigma_e_post[post_i,:])})
            # Evaluate quadrant of covariance matrix associated only with predictions
            x_pred_i = np.concatenate((np.tile(theta_post[post_i,:],(n_pred,1)), x_pred_trans),axis=1)
            cov_pred_i = cov_pred(x_pred_i).eval({sigma_eta : float(sigma_eta_post[post_i]), ls: ls_post[post_i,:]}) +np.eye(n_pred)*1e-8
            # Evaluate cross-covariance between training data and predicitons. Here there is no observation error as no diagonal entries
            cross_cov_i = cov_pred(x_train_i,x_pred_i).eval({sigma_eta : float(sigma_eta_post[post_i]), ls: ls_post[post_i,:]})
            # Use the closed form expressions for making predictions with zero-mean GP
            # Note. It could be possible to use the prediction capability of the pymc gp,
            # but this may need some modification to the mdethod given the different covariance structure
            # between training data and new predictions
            mu_pred[:,i] = np.matmul(cross_cov_i.T,np.linalg.solve(cov_train_i,y_eta))
            cov_sam = cov_pred_i - np.matmul(cross_cov_i.T, np.linalg.solve(cov_train_i, cross_cov_i))
            f_pred[:,i] = np.random.multivariate_normal(mu_pred[:,i], cov_sam)
            sigma_pred[:,i] = np.diagonal(cov_sam)

#------------------------------------------------------------------------------
#           Transform back onto correct scale then plot predictions
#------------------------------------------------------------------------------

    # Convert the mean and standard deviation back onto the true scale
    sigma_pred = np.sqrt(sigma_pred)
    mu_pred = rescale_output(mu_pred, mu_y = mu_eta, sigma_y = sd_eta)
    f_pred = rescale_output(f_pred, mu_y = mu_eta, sigma_y = sd_eta)
    sigma_pred = rescale_output(sigma_pred, sigma_y = sd_eta, std = True)
    sigma_e_post = rescale_output(sigma_e_post, sigma_y = sd_eta, std = True)

    # Take averages across prior and posterior datasets and calculate standard deviations
    prior_mean = np.mean(eta_all_plot,axis=1)
    prior_sd = np.std(eta_all_plot,axis=1)
    posterior_mean = np.mean(f_pred,axis=1)
    posterior_sd = np.std(f_pred,axis=1)
    posterior_e = np.mean(sigma_e_post)

    fig6, axes6 = plt.subplots(1,2)
    axes6[0].plot(x_plot, prior_mean, "r", label = "Prior mean")
    axes6[0].plot(x_plot, prior_mean+2*prior_sd, color = "tab:red", label= "Prior 95% Interval")
    axes6[0].plot(x_plot, prior_mean-2*prior_sd, color = "tab:red")
    axes6[0].fill_between(np.array(x_plot), prior_mean+2*prior_sd, prior_mean-2*prior_sd, alpha=0.25, color = "tab:red")
    axes6[1].plot(x_pred,f_pred[:,0],"-c", linewidth=0.5, label = "Posterior prediction") # Only want legend for first sample to avoid clutter
    axes6[1].plot(x_pred,f_pred[:,1:],"-c", linewidth=0.5)
    axes6[1].plot(x_pred, posterior_mean, "r", label = "Posterior mean")
    axes6[1].plot(x_pred, posterior_mean+2*posterior_sd, color = "tab:red", label= "Posterior 95% Interval")
    axes6[1].plot(x_pred, posterior_mean-2*posterior_sd, color = "tab:red")
    axes6[1].plot(x_pred, posterior_mean+2*posterior_sd+2*posterior_e, color = "m", label= "Observation error 95% Interval")
    axes6[1].plot(x_pred, posterior_mean-2*posterior_sd-2*posterior_e, color = "m")
    axes6[1].fill_between(x_pred.reshape(-1), posterior_mean+2*posterior_sd, posterior_mean-2*posterior_sd, alpha=0.25, color = "tab:red")
    axes6[1].fill_between(x_pred.reshape(-1), posterior_mean+2*posterior_sd+2*posterior_e, posterior_mean-2*posterior_sd-2*posterior_e, alpha=0.25, color = "tab:pink")
    axes6[0].set_title("Prior predictions")
    axes6[1].set_title("Posterior predictions")
    for ax6 in axes6:
        ax6.set_xlabel("x (m)")
        ax6.set_ylabel("displacement (m)")
        ax6.set_xlim([0,L])
        ax6.plot(x_plot, delta_plot,"-b",linewidth=1, label="True response")
        ax6.plot(dt_data.x.values, dt_data.delta.values,"rx",markersize=12, markeredgewidth=2, label="Experimental data")
        ax6.legend()

    plt.show()