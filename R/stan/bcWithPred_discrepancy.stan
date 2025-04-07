// Code for Bayesian calibration of computer models, and subsequent prediction using the calibrated model. For predictions, separate contributions from emulator and discrepency terms are outputted for visualisation purposes
// Modified from https://github.com/adChong/bc-stan

functions {
  // define a function for calculating the covariance matrix for joint field data, simulation data, and predictions with separate 
  // entries for separated components of emulator and discrepancy terms (at the same input values used for calibrated predictions, x_pred)
  matrix cal_pred_disc_cov(int n, int m, int n_pred, int p, int q, matrix xf, row_vector tf, matrix xc, matrix tc, matrix x_pred, row_vector beta_eta, real lambda_eta, row_vector beta_delta, real lambda_delta, real lambda_e){
    // Note: n is the number of observed data points
    //       m is the number of simulated (model) data points
    //       n_pred is the number of predictions
    int N = m + n + 3*n_pred; // Total number of  points (and therefore dimension of covariance matrix)
    matrix[N, N] sigma_z; // declare variable for joint covariance matrix
    {
    // declare variables
    matrix[n+m+2*n_pred, p+q] xt;                   // joint matrix of inputs (both controlled and calibration inputs)
    matrix[n+2*n_pred,p] X;                         // joint matrix of field data and prediction inputs
    matrix[m+n+2*n_pred, m+n+2*n_pred] sigma_eta;   // emulator covariance
    matrix[n+2*n_pred, n+2*n_pred] sigma_delta;     // discrepancy term covariance
    row_vector[p] temp_delta;                       // temporary variable used to determine difference between inputs
    row_vector[p+q] temp_eta;                       // temporary variable used to determine difference between inputs
    
    // Assemble joint matrix of controlled and calibration inputs for observed data, model data, and prediction points
    // Stored in format:
    // xt = [[xf,tf],[xc,tc],[x_pred,tf],[x_pred,tf]]
    // where:
    // xf = controlled inputs at observed field data points
    // tf = calibration inputs for field data - inferred through simulation
    // xc = controlled inputs in Design of Experiments for emulator
    // tc = calibration inputs in Design of Experiments for emulator
    // x_pred = controlled inputs at which predictions are required
    xt[1:n, 1:p] = xf;
    xt[1:n, (p+1):(p+q)] = rep_matrix(tf, n);
    xt[(n+1):(n+m), 1:p] = xc;
    xt[(n+1):(n+m), (p+1):(p+q)] = tc;
    xt[(n+m+1):(n+m+n_pred), 1:p] = x_pred;
    xt[(n+m+n_pred+1):(n+m+2*n_pred), 1:p] = x_pred;
    xt[(n+m+1):(n+m+2*n_pred), (p+1):(p+q)] = rep_matrix(tf, 2*n_pred);

    // Combine field data points and input values at which predictions are required into a single matrix, for calculation of discrepancy covariance
    // note that it is only necessary to evaluate this matrix at n+2*n_pred points, as the emulator training data, and (uncalibrated) emulator predictions, do not depend upon this covariance.
    X[1:n,:] = xf;
    X[(n+1):(n+n_pred),:] = x_pred;
    X[(n+n_pred+1):(n+2*n_pred),:] = x_pred;
    
    // begin to assemble covariance matrix. 
    sigma_eta = diag_matrix(rep_vector((1 / lambda_eta), (n+m+2*n_pred)));
    // off-diagonal elements of sigma_eta
    for (i in 1:(n+m+2*n_pred-1)) {
      for (j in (i+1):n+m+2*n_pred) {
        temp_eta = xt[i] - xt[j];
        sigma_eta[i, j] = beta_eta .* temp_eta * temp_eta';
        sigma_eta[i, j] = exp(-sigma_eta[i, j]) / lambda_eta;
        sigma_eta[j, i] = sigma_eta[i, j];
      }
    }
    
    // now calculate sigma_delta (components of covariance due to model discrepancy)
    sigma_delta = diag_matrix(rep_vector((1 / lambda_delta), (n+2*n_pred)));
    // off-diagonal elements of sigma_delta
    for (i in 1:(n+2*n_pred-1)) {
      for (j in (i+1):(n+2*n_pred)) {
        temp_delta = X[i] - X[j];
        sigma_delta[i, j] = beta_delta .* temp_delta * temp_delta';
        sigma_delta[i, j] = exp(-sigma_delta[i, j]) / lambda_delta;
        sigma_delta[j, i] = sigma_delta[i, j];
      }   
    }
    
    // assemble covariance matrix sigma_z by adding components due to emulator, discrepancy, and observation errors
    // structure of covariance matrix is arranged to match outputs arranged as: [field data, emulator training data, calibrated predictions, emulator predictions, prediction discrepancy]
    sigma_z = rep_matrix(0,N,N);
    sigma_z[1:(n+m+2*n_pred),1:(n+m+2*n_pred)] = sigma_eta;
    sigma_z[1:n, 1:n] = sigma_eta[1:n, 1:n] + sigma_delta[1:n, 1:n];
    sigma_z[1:n, (n+m+1):(n+m+n_pred)] = sigma_eta[1:n, (n+m+1):(n+m+n_pred)] + sigma_delta[1:n, (n+1):(n+n_pred)];
    sigma_z[1:n, (n+m+2*n_pred+1):N] = sigma_delta[1:n,(n+n_pred+1):(n+2*n_pred)];
    sigma_z[(n+m+1):(n+m+n_pred), 1:n] = sigma_eta[(n+m+1):(n+m+n_pred), 1:n] + sigma_delta[(n+1):(n+n_pred),1:n];
    sigma_z[(n+m+1):(n+m+n_pred), (n+m+1):(n+m+n_pred)] = sigma_eta[(n+m+1):(n+m+n_pred), (n+m+1):(n+m+n_pred)] + sigma_delta[(n+1):(n+n_pred), (n+1):(n+n_pred)];
    sigma_z[(n+m+1):(n+m+n_pred),(n+m+2*n_pred+1):N] = sigma_delta[(n+1):(n+n_pred),(n+n_pred+1):(n+2*n_pred)];
    sigma_z[(n+m+2*n_pred+1):N,1:n] = sigma_delta[(n+n_pred+1):(n+2*n_pred),1:n];
    sigma_z[(n+m+2*n_pred+1):N,(n+m+1):(n+m+n_pred)] = sigma_delta[(n+n_pred+1):(n+2*n_pred),(n+1):(n+n_pred)];
    sigma_z[(n+m+2*n_pred+1):N,(n+m+2*n_pred+1):N] = sigma_delta[(n+n_pred+1):(n+2*n_pred),(n+n_pred+1):(n+2*n_pred)];
    // add observation errors
    for (i in 1:n) {
      sigma_z[i, i] = sigma_z[i, i] + (1.0 / lambda_e);
    }
  }
  return sigma_z;
}

  // Define a function for analytical expressions for GP posterior predictions for a zero-mean prior, with covariance structure as constructed using cal_pred_disc_cov, 
  // for a calibrated Gaussian process which outputs separate calibrated predictions, and the corresponding discrepancy and emulator contributions
  vector gp_pred_disc_rng(int m, int n, int n_pred, vector y1, int p, int q, matrix xf, row_vector tf, matrix xc, matrix tc, matrix x_pred, row_vector beta_eta, real lambda_eta, row_vector beta_delta, real lambda_delta, real lambda_e) {
    int N1 = m + n;     // number of training data points
    int N2 = 3*n_pred;  // number of points at which predictions are required
    vector[N2] f_pred;  // initialise function output 
    {
    matrix[(N1+N2), (N1+N2)] K_all; // Joint covariance matrix of training and test data points
    matrix[N1, N1] L;               // Cholesky decomposition of covariance matrix
    vector[N1] L_div_y;             // 
    vector[N1] K_div_y;
    vector[N2] f_mu;                // Posterior predictive GP mean at x_pred
    matrix[N1, N2] v_pred;
    matrix[N2, N2] f_cov;           // Posterior predictive GP covariance
    
    // Calculate joint covariance matrix of training and test data points
    K_all = cal_pred_disc_cov(n, m, n_pred, p, q, xf, tf, xc, tc, x_pred, beta_eta, lambda_eta, beta_delta, lambda_delta, lambda_e);
    // Efficiently calculate the posterior predictive GP mean and covariance using the Cholesky decomposition of the covariance matrix
    // For expressions, see Chapter 2, "Gaussian Processes for Machine Learning", Rasmussen and Williams, MIT Press, 2006, ISBN 0-262-18253-X.
    // http://www.gaussianprocess.org/gpml/
    // Isolate portion of the covariance of field data and model output data
    L = cholesky_decompose(K_all[1:N1,1:N1]);
    L_div_y = mdivide_left_tri_low(L, y1);
    K_div_y = mdivide_right_tri_low(L_div_y', L)';
    // Calculate posterior predictive mean through matrix algebra
    f_mu = K_all[1:N1,(N1+1):(N1+N2)]' * K_div_y;
    v_pred = mdivide_left_tri_low(L, K_all[1:N1,(N1+1):(N1+N2)]);
    // calcluate posterior predictive covariance through matrix algebra
    // note: a small nugget term is added to this to force the matrix to be positive-semi-definite when many predictions are required
    f_cov = K_all[(N1+1):(N1+N2),(N1+1):(N1+N2)] - v_pred'*v_pred + diag_matrix(rep_vector(1e-8,N2));
    // sample predictions from a Gaussian process (discretised as multivariate normal) using posterior predictive mean and covariance
    f_pred = multi_normal_rng(f_mu, f_cov);
    }
    return f_pred;
}

}

data {
  int<lower=1> n;       // number of field data points
  int<lower=1> m;       // number of computer simulation points
  int<lower=1> n_pred;  // number of predictions which are required
  int<lower=1> p;       // number of observable/controlled inputs, x
  int<lower=1> q;       // number of calibration parameters, t
  vector[n] y;          // field observations
  vector[m] eta;        // output of computer simulations
  matrix[n, p] xf;      // observable/controlled inputs corresponding to y
  
  // (xc, tc): design points (controlled and calibration inputs) corresponding to eta
  matrix[m, p] xc; 
  matrix[m, q] tc; 
  // x_pred: new design points for predictions
  matrix[n_pred, p] x_pred; 
}

transformed data {
  vector[n+m] y_eta;
  vector[n+m] mu; // prior mean vector
  // set the prior mean vector to zero
  for (i in 1:(m+n)) {
    mu[i] = 0;
  }
  // Combine y (field data) and eta (model data) into one vector for simulation purposes
  y_eta = append_row(y, eta); // y_eta = [y, eta]
}

parameters {
  // Declare parameters which are to be inferred through simulation
  // tf: calibration parameters
  // rho_eta: reparameterisation of correlation length, beta_eta
  // rho_delta: reparameterisation of correlation length, beta_delta
  // lambda_eta: precision parameter for emulator
  // lambda_delta: precision parameter for discrepancy term
  // lambda_e: precision parameter of observation error
  row_vector<lower=0, upper=1>[q] tf; 
  row_vector<lower=0, upper=1>[p+q] rho_eta; 
  row_vector<lower=0, upper=1>[p] rho_delta; 
  real<lower=0> lambda_eta; 
  real<lower=0> lambda_delta;
  real<lower=0> lambda_e;
}

transformed parameters {
  // beta_delta: correlation parameter for discrepancy term
  // beta_e: correlation parameter of observation error
  row_vector[p+q] beta_eta;
  row_vector[p] beta_delta;
  // cacluate correlation length from transformed parameters rho_eta and rho_delta
  beta_eta = -4.0 * log(rho_eta);
  beta_delta = -4.0 * log(rho_delta);
}

model {
  // declare variables
  matrix[(n+m), (n+m)] sigma_eta;   // simulator covariance
  matrix[n, n] sigma_delta;         // discrepancy term covariance
  matrix[(n+m), (n+m)] sigma_z;     // overall covariance matrix
  matrix[(n+m), (n+m)] L;           // cholesky decomposition of covariance matrix
  
  // Now calculate covariance matrix for joint matrix of observed field data and model simulation data
  // Note that this omits the prediction points for computational efficiency, as these are not required for the inverse problem of determining calibration parameters.
  sigma_z = cal_pred_disc_cov(n, m, 0, p, q, xf, tf, xc, tc, rep_matrix(0,0,p), beta_eta, lambda_eta, beta_delta, lambda_delta, lambda_e);
  
  // Specify priors here
  rho_eta[1:(p+q)] ~ beta(1.0, 0.3);    // correlation parameter for emulator
  rho_delta[1:p] ~ beta(1.0, 0.3);      // correlation parameter for discrepancy
  lambda_eta ~ gamma(10, 10);           // precision parameter for emulator, gamma (shape, rate)
  lambda_delta ~ gamma(10, 0.3);        // precision parameter for discrepancy
  lambda_e ~ gamma(10, 0.03);           // precision parameter for observation error (1/standard deviation)

  L = cholesky_decompose(sigma_z); // cholesky decomposition of covariance
  
  // Tell Stan that y and eta are from a multivariate normal distribution with mean mu, and covariance sigma_z.
  // i.e. are field data and model outputs are assumed to be drawn from a Gaussian process with the specified covariance structure
  y_eta ~ multi_normal_cholesky(mu, L);
}

generated quantities {

  // This is the block of code in which the predictions, emulator terms, and discrepancy values are generated
  
  // declare variables
  vector[3*n_pred] f_eta_delta_pred;    // joint vector of predictions, emulator and discrepancy values
  vector[n_pred] f_pred;                // calibrated predictions
  vector[n_pred] eta_pred;              // emulator values
  vector[n_pred] delta_pred;            // discrepancy values
  
  // use custom function to sample from GP posterior predictive distribution using the inferred emulator hyperparameters
  // this outputs a joint vector of calibratied predictions, emulator, and discrepancy values
  f_eta_delta_pred = gp_pred_disc_rng(m,n,n_pred,y_eta,p, q, xf, tf, xc, tc, x_pred, beta_eta, lambda_eta, beta_delta, lambda_delta, lambda_e);
  // separate the different output quantities
  f_pred = f_eta_delta_pred[1:n_pred];
  eta_pred = f_eta_delta_pred[n_pred+1:2*n_pred];
  delta_pred = f_eta_delta_pred[2*n_pred+1:3*n_pred];
}

