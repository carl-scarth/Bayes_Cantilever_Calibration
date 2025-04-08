  # Set up R
  
  library(data.table)
  library(rstan)
  library(maximin)
  library(matrixStats)
  library(colormap)
  
  # Set current working directory. This should be modified to match the directory
  # at which the stan code and any data is stored
  setwd("C:/Users/cs2361/Documents/R/Gaussian Processes")
  
  # include functions which are called in this code
  source("cantileverBeam.R")
  source("estimate_mode.R")
  
  #-------------------------------------------------------------------------------
  
  # Set up model parameters
  
  # First we set up the parameters
  n_data = 10          # number of experimental data points
  n_repeats = 3        # number of repeated data points
  n_simulations = 30   # number of simulations
  N_sim_data = 30      # number of output points in each simulation
  N_plot = 100         # Number of points at which beam deflection is plotted
  n_maximin = 30       # Number of model output data points which are to be retained for training the emulator. I use a smaller number here as the code runs very slowly if I use everything
  
  # Define nominal parameters of underlying model, which are used to generate the synthetic data.
  
  P <- 10 # Applied load (N)
  L <- 1
  E <- 68E9 # Young's modulus (Aluminium)
  d <- 0.01 # depth
  b <- 0.04 # breadth
  
  # Define magnitude of the synthetic noise which will be added to model outputs to get "observed" data
  sigma_e = 0.00025 # standard deviation
  sigma2_e = sigma_e^2  # variance
    
  #-------------------------------------------------------------------------------
  
  # generate deviation from nominal beam depth used to determine "true" underlying beam depth
  # This is currently commented out for reproducability, but can be uncommented if desired
  #d_data = data.table(matrix(runif(n = 1, min = -0.1*d, max = 0.1*d), nrow = n_beam))
  
  # Pre-define deviation from nominal beam thickness based upon previously generated data
  # (comment out if using line 40 to generate new synthetic data)
  d_data = data.table(matrix(0.0005782702)) # pre-define for reproducability
  colnames(d_data) = c("d")
  
  # Define coordinates at which displacements are observed
  # (not including 0 and L, assuming it is difficult to take measurements near boundary)
  x = seq(0.05*L, 0.95*L, length.out = n_data)
  # add some repeated observations. These arise at n_repeats randomly selected coordinates from x
  x_repeat = x[ceiling(runif(n=n_repeats,min=0,max=n_data))]
  x = c(x,x_repeat) # code this properly in the actual version....
  dt_coords = data.table(x)
  
  # Calculate displacement at these points, given the "true" value of depth, using the separate "cantileverBeam.R" function
  delta = beamDeflection(x,E,b,d + as.numeric(d_data),P)
  dt_data = cbind(dt_coords, data.frame("D" = delta))
  
  # Generate synthetic, "observed" data by adding iid noise to the "true" displacement
  dt_data$data = dt_data$D + rnorm(nrow(dt_data), mean = 0, sd = sigma_e)
  
  # Use the below line to write observed data to a csv file if needed later
  #write.csv(dt_data, "benchmark_observations.csv", row.names = FALSE)
  
  # Use the below line to load previously generated observed data from a csv file, if needed to reproduce past results
  # dt_data = read.table("benchmark_observations.csv", header = TRUE, sep=',')
  
  #-------------------------------------------------------------------------------
  
  # Plot observed displacements
  
  dev.new(noRStudioGD = TRUE) # create new window in which to plot
  
  # define points at which the "true" displacement is to be plotted
  x_plot = seq(0,L,length.out = 100)
  # run beam model to determine "true" displacement at these points
  defplot = beamDeflection(x_plot,E,b,d + as.numeric(d_data),P)
  # plot the "observed" data
  plot(dt_data$x,dt_data$data,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  # plot the "true" displacements
  lines(x_plot,defplot,"lwd"=2,"col"="blue")
  # plot legend
  legend(x = "topright",legend = c("true response","observed response"),col=c("blue","red"),pch=c(NA,4),lty = c(1,NA),lwd = c(2,2))
  
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
  
  # In this section of code the data is put into the correct format for input to stan,
  # and the stan code for calibration, and prediction, is run
  
  # store a set of x coordinates for plotting calibration predictions
  x_pred = as.matrix(x_plot)
  
  # Set up the environment for the Stan model to run in parallel. Taken from:
  # https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#21_Simulating_From_A_Gaussian_Process
  
  # set stan to execute multiple Markov chains in parallel
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  parallel:::setDefaultClusterOptions(setup_strategy = "sequential")
  util = new.env()
  par(family="CMU Serif", las=1, bty="l", cex.axis=1, cex.lab=1, cex.main=1,
      xaxs="i", yaxs="i", mar = c(5, 5, 3, 5))
  
  # Note, the below calibration is a modification of the implementation at:
  # https://github.com/adChong/bc-stan/blob/master/src/main.R
  # Specifically, a much more efficient code for making predictions with the emulator has been implemented
  
  # get dimensions of the various datasets
  p = ncol(dt_coords)         # number of input factors (x)
  q = ncol(XT_sim) - p        # number of calibration parameters (theta, or d in this example)
  n = nrow(dt_data)           # sample size of observed field data
  m = nrow(XT_sim)            # sample size of computer simulation data
  n_pred = nrow(x_pred)       # number of predictions
  
  # extract data from the various tables
  y = dt_data$data                # "observed" displacements
  eta = dt_simulation$D           # simulation output (emulator training data)
  xf = as.matrix(dt_data$x)       # "observed" (controllable) input
  xc = as.matrix(XT_sim$x_sim)    # simulation (controllable) input (emulator training data)
  tc = as.matrix(XT_sim$d)        # simulation (uncontrolled) input (calibration parameter)
  
  # standardise ouputs to have zero sample mean unit sample variance. 
  eta_mu = mean(as.matrix(eta), na.rm = TRUE) # mean value
  eta_sd = sd(as.matrix(eta), na.rm = TRUE)   # standard deviation
  eta = (eta - eta_mu)/eta_sd
  # note that y is normalised using the same parameters as eta for consistency
  y = (y - eta_mu)/eta_sd
  
  # Standardise controlled and uncontrolled inputs to lie on [0,1]
  # start with controlled inputs x
  x = rbind(as.matrix(xf), as.matrix(xc))
  x_min = as.matrix(colMins(x))
  x_max = as.matrix(colMaxs(x))
  xf = (xf - matrix(rep(x_min,n,ncol=p,byrow=TRUE)))/matrix(rep(x_max-x_min,n,ncol=p,byrow=TRUE))
  xc = (xc - matrix(rep(x_min,m,ncol=p,byrow=TRUE)))/matrix(rep(x_max-x_min,m,ncol=p,byrow=TRUE))
  x_pred = (x_pred - matrix(rep(x_min,n_pred,ncol=p,byrow=TRUE)))/matrix(rep(x_max-x_min,n_pred,ncol=p,byrow=TRUE))
  # now also standardise calibration inputs
  t_min = as.matrix(colMins(tc))
  t_max = as.matrix(colMaxs(tc))
  tc = (tc - matrix(rep(t_min,m,ncol=q,byrow=TRUE)))/matrix(rep(t_max-t_min,m,ncol=q,byrow=TRUE))
  
  # create data as list for input to Stan
  stan_data = list(n=n, m=m, n_pred=n_pred, p=p, y=c(as.matrix(y)), q=q, eta=c(as.matrix(eta)), 
                   xf=as.matrix(xf), xc=as.matrix(xc), 
                   x_pred=as.matrix(x_pred), tc=as.matrix(tc))
  
  
  # Run code for Bayesian Calibration, and subsequent prediction using Gaussian process
  fit0 = stan(file = "bcWithPred_discrepancy.stan",
             data = stan_data,
             iter = 4000,
             chains = 3)
  
  #-------------------------------------------------------------------------------
  
  # This section of code for post-processing, and generation of plots of 
  # calibrated model
  
  # plot traceplots, excluding warm-up
  dev.new(noRStudioGD = TRUE)  # generate plots in separate window
  stan_trace(fit0, pars = c("tf", "rho_eta", "rho_delta","lambda_eta", "lambda_delta","lambda_e"))
  
  # summarise results to check convergence
  print(fit0, pars = c("tf", "beta_eta", "beta_delta","lambda_eta", "lambda_delta","lambda_e","f_pred","eta_pred","delta_pred"))
  
  # extract samples from stan output
  samples <- extract(fit0)
  N_samples = dim(samples$rho_eta)[1] # get total number of samples
  
  # extract predicted displacements and transform these onto their correct scale
  # (recalling that observations were transformed to have zero sample mean, and 
  # unit sample variance)
  f_pred = samples$f_pred
  f_trans = f_pred*eta_sd + eta_mu
  # also transform emulator and discrepancy values onto their correct scales
  eta_pred = samples$eta_pred
  eta_trans = eta_pred*eta_sd + eta_mu
  delta_pred = samples$delta_pred
  delta_trans = delta_pred*eta_sd
  
  # Produce plots of posterior and prior distributions of correlation lengths
  # (rho), and marginal precisions (lambda)
  dev.new(noRStudioGD = TRUE)
  par(mfrow = c(2,3))
  
  # start with emulator (virtual response) correlation lengths
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
  # plot correlation length of discrepancy
  # plot histogram of posterior distribution
  hist(samples$rho_delta[,1],
       main = "correlation length discrepancy",
       xlab = "rho_delta1",
       col = "firebrick1",
       breaks = 25,
       freq = FALSE,
       xlim = c(0,1),
       cex.lab=1.5,
       cex.axis=1.5)
  # Plot prior distribution
  prior_plot = dbeta(rho_plot,shape1=1,shape2=0.3)
  lines(rho_plot,prior_plot,lwd=3,col="blue")
  
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
  
  # marginal precision of discrepancy
  # plot posterior
  hist(samples$lambda_delta,
       main = "lambda_delta",
       xlab = "lambda_delta",
       col = "firebrick1",
       breaks = 25,
       freq = FALSE,
       xlim = c(0,75),
       cex.axis=1.5,
       cex.lab=1.5)
  # plot prior
  lambda_plot = seq(0,75, length.out = 100)
  prior_plot = dgamma(lambda_plot,shape=10,rate=0.3)
  lines(lambda_plot,prior_plot,lwd=3,col="blue")
  
  # marginal precision of observation error
  #plot posterior
  hist(samples$lambda_e,
       main = "lambda_epsilon",
       xlab = "lambda_epsilon",
       col = "firebrick1",
       breaks = 25,
       freq = FALSE,
       xlim = c(0,750),
       cex.axis=1.5,
       cex.lab=1,5)
  # plot prior
  lambda_plot = seq(0,750, length.out = 100)
  prior_plot = dgamma(lambda_plot,shape=10,rate=0.03)
  lines(lambda_plot,prior_plot,lwd=3,col="blue")
  
  # Before plotting their distributions, the calibration parameter (depth) must
  # be transformed back onto its original scale, recalling that this was 
  # initially transformed onto [0,1]
  tf_trans = samples$tf
  tf_trans = tf_trans*matrix(rep(t_max-t_min,N_samples,ncol=q,byrow=TRUE)) + matrix(rep(t_min,N_samples,ncol=q,byrow=TRUE))
  
  # Plot prior and posterior distribution of calibration parameter (depth)
  dev.new(noRStudioGD = TRUE) # plot in new window
  # plot histogram of posterior
  hist((d+tf_trans), 
         main = "Calibration Parameter",
         xlab = "depth (m)",
         col = "brown1",
         breaks = 25,
         freq = FALSE,
         xlim = c(0.9*d,1.1*d),
         cex.lab = 1.25,
         cex.axis = 1.25)
  # overlay plot of prior distribution
  d_plot = seq(-0.1*d,0.1*d, length.out = 100)
  prior_plot = dunif(d_plot,min=-0.1*d,max=0.1*d,log=FALSE)
  lines(d_plot+d,prior_plot,lwd=3,"col"="blue")
  
  # estimate mode of the posterior distribution (calling estimate mode function)
  modes = estimate_mode(tf_trans)+d # add onto nominal value of d, as calibration is performed on the deviation from this value
  print("calibration parameter mode = ")
  print(modes)
  
  # Generate plots of samples of the calibrated Gaussian process predictive model, alonside corresponding plots of the emulator and discrepancy value
  dev.new(noRStudioGD = TRUE) # plot in new window
  par(mfrow = c(1,3))
  # Don't plot the full set of samples as it gets messy
  plot_int = 200 # interval between Gaussian process samples which are plotted
  plot_idx <- seq(1, nrow(f_pred), plot_int) # index of Gaussian process samples to plot
  
  # Define colourmap for plot. Note, colours taken from:
  # https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#21_Simulating_From_A_Gaussian_Process
  nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C", "#A25050", "#8F2727", "#7C0000")
  line_colors <- colormap(colormap=nom_colors, nshades=length(plot_idx))
  # plot each of the indexed samples of the calibrated predictions
  plot(x_plot,f_trans[plot_idx[1],],type="l",col=line_colors[1],lwd=1.75,xlim=c(0,L),ylim=c(min(f_trans),max(f_trans)),'xlab'="x (m)",'ylab'="displacement (m)",cex.lab=1.5,cex.axis=1.5,main = 'Calibrated Gaussian Process Samples')
  for (n in 2:length(plot_idx)){
    lines(x_plot, f_trans[plot_idx[n],], col=line_colors[n],lwd=1.75)
  }
  # Overlay plots of the observed data, and the underlying "true" response
  points(dt_data$x,dt_data$data,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,defplot,"lwd"=2,"col"="blue")
  # plot legend
  legend(x = "bottomleft",legend = c("sample response","true response","observed response"),col=c(nom_colors[6],"blue","red"),pch=c(NA,NA,4),lty = c(1,1,NA),lwd = c(1.75,2,2))

  # plot the same samples of the emulator without the discrepancy
  plot(x_plot,eta_trans[plot_idx[1],],type="l",col=line_colors[1],lwd=1.75,xlim=c(0,L),ylim=c(min(f_trans),max(f_trans)),'xlab'="x (m)",'ylab'="displacement (m)",cex.lab=1.5,cex.axis=1.5,main = 'Emulator Samples')
  for (n in 2:length(plot_idx)){
    lines(x_plot, eta_trans[plot_idx[n],], col=line_colors[n],lwd=1.75)
  }
  # Overlay plots of the observed data, and the underlying "true" response
  points(dt_data$x,dt_data$data,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,defplot,"lwd"=2,"col"="blue")
  
  # Now plot discrepancy terms
  plot(x_plot,delta_trans[plot_idx[1],],type="l",col=line_colors[1],lwd=1.75,xlim=c(0,L),ylim=c(min(delta_trans),max(delta_trans)),'xlab'="x (m)",'ylab'="discrepancy (m)",cex.lab=1.5,cex.axis=1.5,main = 'Discrepancy Samples')
  for (n in 2:length(plot_idx)){
    lines(x_plot, delta_trans[plot_idx[n],], col=line_colors[n],lwd=1.75)
  }
  # Overlay plots of distance of the observed points from their "true" value, as well as the "true" discrepancy of zero (as data is synthetic)
  points(dt_data$x,dt_data$data-dt_data$D,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,rep(0,length(x_plot)),"lwd"=2,"col"="blue")

  
  # plot average calibrated prediction (effectively integrating out uncertainty
  # across emulator hyperparameters, and the calibration parameter)
  # each column is a point in space, so we can use colMeans to give the average prediction at each point in space
  f_trans_mu = colMeans(f_trans)
  eta_trans_mu = colMeans(eta_trans)
  delta_trans_mu = colMeans(delta_trans)
  dev.new(noRStudioGD = TRUE) # plot in a new window
  par(mfrow = c(1,3))
  plot(x_plot,f_trans_mu,type="l",col="magenta",main = "Calibrated Model", lwd=2,xlim=c(0,L),ylim=c(min(f_trans),max(f_trans)),'xlab'="x (m)",'ylab'="displacement (m)",cex.lab=1.5,cex.axis=1.5)
  # overlay plots of observed response and "true" underlying response
  points(dt_data$x,dt_data$data,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,defplot,"lwd"=2,"col"="blue")
  # plot legend
  legend(x = "bottomleft",legend = c("calibrated model","true response","observed response"),col=c("magenta","blue","red"),pch=c(NA,NA,4),lty = c(1,1,NA),lwd = c(2,2,2))
  
  # Repeat for plot of emulator
  plot(x_plot,eta_trans_mu,type="l",col="magenta",main = "Emulator Mean", lwd=2,xlim=c(0,L),ylim=c(min(f_trans),max(f_trans)),'xlab'="x (m)",'ylab'="displacement (m)",cex.lab=1.5,cex.axis=1.5)
  # overlay plots of observed response and "true" underlying response
  points(dt_data$x,dt_data$data,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,defplot,"lwd"=2,"col"="blue")
  
  # Finally, plot the average discrepancy
  plot(x_plot,delta_trans_mu,type="l",col="magenta",main = "Discrepancy Mean", lwd=2,xlim=c(0,L),ylim=c(min(delta_trans),max(delta_trans)),'xlab'="x (m)",'ylab'="discrepancy (m)",cex.lab=1.5,cex.axis=1.5)
  # overlay plots of observed response and "true" underlying response
  points(dt_data$x,dt_data$data - dt_data$D,"type"="p","col"="red","pch"=4,"lwd"=2,cex=1,'xlab'='x coordinate (m)','ylab'="displacement (m)",cex.axis=1.3,cex.lab=1.5,'xlim'=c(0,L))
  lines(x_plot,rep(0,length(x_plot)),"lwd"=2,"col"="blue")
  
  # generate plot of quantiles and median. Code also taken from:
  # https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#21_Simulating_From_A_Gaussian_Process
  # define colours used to shade the different quantiles
  c_light <- c("#DCBCBC")
  c_light_highlight <- c("#C79999")
  c_mid <- c("#B97C7C")
  c_mid_highlight <- c("#A25050")
  # define quantiles of interest
  probs = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  # estimate the specified quantiles for the calibrated Gaussian process samples
  cred <- sapply(1:n_pred, function(n) quantile(f_trans[,n], probs=probs))
  # Plot the quantiles and median in a new window
  dev.new(noRStudioGD = TRUE)
  par(mfrow = c(1,3))
  plot(1, type="n", main="Calibrated Model Quantiles and Median", xlab="x (m)", ylab="displacement (m)", xlim=c(0, L), ylim=c(min(f_trans), max(f_trans)))
  # plot the quantiles by overlaying polygons on top of one another
  polygon(c(x_plot, rev(x_plot)), c(cred[1,], rev(cred[9,])), col = c_light, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[2,], rev(cred[8,])), col = c_light_highlight, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[3,], rev(cred[7,])), col = c_mid, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[4,], rev(cred[6,])), col = c_mid_highlight, border = NA)
  # plot the median
  lines(x_plot, cred[5,], col="red", lwd=2)
  # overlay with plots of the true response and observed response
  lines(x_plot,defplot,col="blue",lwd = 3)
  points(dt_data$x,dt_data$data,col = "red",pch=c(4),lwd=2,cex=1)
  # plot legend
  legend(x = "bottomleft",legend = c("median","10th-90th percentile","20th-80th percentile","30th-70th percentile","40th-60th percentile","true response","observed response"),col=c("red",c_light,c_light_highlight,c_mid,c_mid_highlight,"blue","red"),pch=c(NA,NA,NA,NA,NA,NA,4),lty = c(1,1,1,1,1,1,NA),lwd = c(2,2,2,2,2,2,2))
  
  # repeat for emulator
  plot(1, type="n", main="Emulator Quantiles and Median", xlab="x (m)", ylab="displacement (m)", xlim=c(0, L), ylim=c(min(f_trans), max(f_trans)))
  cred <- sapply(1:n_pred, function(n) quantile(eta_trans[,n], probs=probs))
  polygon(c(x_plot, rev(x_plot)), c(cred[1,], rev(cred[9,])), col = c_light, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[2,], rev(cred[8,])), col = c_light_highlight, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[3,], rev(cred[7,])), col = c_mid, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[4,], rev(cred[6,])), col = c_mid_highlight, border = NA)
  # plot the median
  lines(x_plot, cred[5,], col="red", lwd=2)
  # overlay with plots of the true response and observed response
  lines(x_plot,defplot,col="blue",lwd = 3)
  points(dt_data$x,dt_data$data,col = "red",pch=c(4),lwd=2,cex=1)
  
  # finally plot quantiles of the discrepancy
  plot(1, type="n", main="Discrepancy Quantiles and Median", xlab="x (m)", ylab="discrepancy (m)", xlim=c(0, L), ylim=c(min(delta_trans), max(delta_trans)))
  cred <- sapply(1:n_pred, function(n) quantile(delta_trans[,n], probs=probs))
  polygon(c(x_plot, rev(x_plot)), c(cred[1,], rev(cred[9,])), col = c_light, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[2,], rev(cred[8,])), col = c_light_highlight, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[3,], rev(cred[7,])), col = c_mid, border = NA)
  polygon(c(x_plot, rev(x_plot)), c(cred[4,], rev(cred[6,])), col = c_mid_highlight, border = NA)
  # plot the median
  lines(x_plot, cred[5,], col="red", lwd=2)
  # overlay with plots of the true response and observed response
  lines(x_plot,rep(0,length(x_plot)),col="blue",lwd = 3)
  points(dt_data$x,(dt_data$data-dt_data$D),col = "red",pch=c(4),lwd=2,cex=1)