// Code for Bayesian calibration of computer models, and subsequent prediction using the calibrated model
// Modified from https://github.com/adChong/bc-stan
// The main modifications are for making efficient predictions by sampling from closed-form expressions for the posterior predictive distribution  

functions {
  // Define a function which calculates the covariance matrix for joint observed field data, model data, and predictions
  // Note: this is defined here as a separate function to prevent the entire covariance matrix being outputted from Stan, which causes an error if many predictions are required
    matrix cal_pred_cov(int n, int m, int N, int n_pred, int p, int q, matrix xf, row_vector tf, matrix xc, matrix tc, matrix x_pred, row_vector beta_eta, real lambda_eta, row_vector beta_delta, real lambda_delta, real lambda_e){
    // initialise function output
    // note: n is the number of observed data points
    //       m is the number of simulated (model) data points
    //       n_pred is the number of predictions
    //       N = n + m + n_pred
    matrix[N, N] sigma_z; // declare variable for joined covariance matrix
    {
    //main body of the funtion
    // declare variables
    matrix[N, p+q] xt;                      // joint matrix of inputs (both controlled and calibration inputs)
    matrix[n+n_pred, p] X;                  // joint matrix of field data and prediction inputs
    matrix[N, N] sigma_eta;                 // simulator covariance
    matrix[n+n_pred, n+n_pred] sigma_delta; // discrepancy term covariance
    row_vector[p] temp_delta;               // temporary variable used to determine difference between inputs
    row_vector[p+q] temp_eta;               // temporary variable used to determine difference between inputs
    
    // Assemble joint matrix of controlled and calibration inputs for observed data, model data, and prediction points
    // Stored in format:
    // xt = [[xf,tf],[xc,tc],[x_pred,tf]]
    // where:
    // xf = controlled inputs at observed field data points
    // tf = calibration inputs for field data - inferred through simulation
    // xc = controlled inputs in Design of Experiments for emulator
    // tc = calibration inputs in Design of Experiments for emulator
    // x_pred = controlled inputs at which predictions are required
    // note that the calibration input at all points in field data, and those at which predictions are required, are assumed to have the same value, tf, which is sampled in the simulation
    xt[1:n, 1:p] = xf;
    xt[1:n, (p+1):(p+q)] = rep_matrix(tf, n);
    xt[(n+1):(n+m), 1:p] = xc;
    xt[(n+1):(n+m), (p+1):(p+q)] = tc;
    xt[(n+m+1):N, 1:p] = x_pred;
    xt[(n+m+1):N, (p+1):(p+q)] = rep_matrix(tf, n_pred);
    
    // Combine field data points and input values at which predictions are required into a single matrix, for calculation of discrepancy covariance
    X = append_row(xf, x_pred);
    
    // begin to assemble covariance matrix. 
    // Start with sigma_eta (components of covariance due to emulator)
    // Note that this depends on both controlled (x), and calibration inputs (t)
    // diagonal elements of sigma_eta
    sigma_eta = diag_matrix(rep_vector((1 / lambda_eta), N));
    // off-diagonal elements of sigma_eta
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        temp_eta = xt[i] - xt[j];
        sigma_eta[i, j] = beta_eta .* temp_eta * temp_eta';
        sigma_eta[i, j] = exp(-sigma_eta[i, j])/lambda_eta;
        sigma_eta[j, i] = sigma_eta[i, j];
      }
    }

    // now calculate sigma_delta (components of covariance due to model discrepancy)
    // Note that this only depends upon controlled inputs, x
    // diagonal elements of sigma_delta
    sigma_delta = diag_matrix(rep_vector((1 / lambda_delta), n+n_pred));
    // off-diagonal elements of sigma_delta
    for (i in 1:(n+n_pred-1)) {
      for (j in (i+1):(n+n_pred)) {
        temp_delta = X[i] - X[j];
        sigma_delta[i, j] = beta_delta .* temp_delta * temp_delta';
        sigma_delta[i, j] = exp(-sigma_delta[i, j]) / lambda_delta;
        sigma_delta[j, i] = sigma_delta[i, j];
      }   
    }
    
    // assemble covariance matrix sigma_z by adding components due to emulator, discrepancy, and observation errors
    // structure of covariance matrix is arranged to match the appropriate components of xt as outlined above
    sigma_z = sigma_eta;
    sigma_z[1:n, 1:n] = sigma_eta[1:n, 1:n] + sigma_delta[1:n, 1:n];
    sigma_z[1:n, (n+m+1):N] = sigma_eta[1:n, (n+m+1):N] + sigma_delta[1:n, (n+1):(n+n_pred)];
    sigma_z[(n+m+1):N, 1:n] = sigma_eta[(n+m+1):N, 1:n] + sigma_delta[(n+1):(n+n_pred),1:n];
    sigma_z[(n+m+1):N, (n+m+1):N] = sigma_eta[(n+m+1):N, (n+m+1):N] + sigma_delta[(n+1):(n+n_pred), (n+1):(n+n_pred)];
    // add observation errors
    for (i in 1:n) {
      sigma_z[i, i] = sigma_z[i, i] + (1.0/lambda_e);
    }
  }
  return sigma_z;
}

}

data {
  int<lower=1> n;      // number of field data points
  int<lower=1> m;      // number of computer simulation points
  int<lower=1> n_pred; // number of predictions which are required
  int<lower=1> p;      // number of observable/controlled inputs, x
  int<lower=1> q;      // number of calibration parameters, t
  vector[n] y;         // field observations
  vector[m] eta;       // output of computer simulations
  matrix[n, p] xf;     // observable/controlled inputs corresponding to y
  // (xc, tc): design points (controlled and calibration inputs) corresponding to eta
  matrix[m, p] xc; 
  matrix[m, q] tc; 
  // x_pred: new design points for predictions
  matrix[n_pred, p] x_pred; 
}

transformed data {
  int<lower = 1> N;
  vector[n+m] y_eta;
  vector[n+m] mu; // mean vector
  
  // Total number of  points (and therefore dimension of covariance matrix)
  N = n + m + n_pred;
  
  // set the mean vector to zero
  mu = rep_vector(0, n+m);
  
  // Combine y (field data) and eta (model data) into one vector for simulation purposes
  y_eta = append_row(y, eta); // y_eta = [y, eta]
}

parameters {
  // Declare parameters which are to be inferred through simulation
  // tf: calibration parameters
  // rho_eta: reparameterisation of beta_eta
  // rho_delta: reparameterisation of beta_delta
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
  beta_eta = -4.0 * log(rho_eta);
  beta_delta = -4.0 * log(rho_delta);
}

model {
  // declare variables
  matrix[(n+m), (p+q)] xt;        // joint matrix of inputs (both controlled and calibration inputs)
  matrix[(n+m), (n+m)] sigma_eta; // simulator covariance
  matrix[n, n] sigma_delta;       // discrepancy term covariance
  matrix[(n+m), (n+m)] sigma_z;   // overall covariance matrix
  matrix[(n+m), (n+m)] L;         // cholesky decomposition of covariance matrix 
  row_vector[p] temp_delta;       // temporary variable used to determine difference between inputs
  row_vector[p+q] temp_eta;       // temporary variable used to determine difference between inputs

  // Assemble joint matrix of controlled and calibration inputs for observed field data and model data
  // Stored in format:
  // xt = [[xt,tf],[xc,tc]]
  // note that the calibration input at all points in field data are assumed to have the same value, tf, which is sampled in the simulation
  xt[1:n, 1:p] = xf;
  xt[(n+1):(n+m), 1:p] = xc;
  xt[1:n, (p+1):(p+q)] = rep_matrix(tf, n);
  xt[(n+1):(n+m), (p+1):(p+q)] = tc;

  // Now calculate covariance matrix for joint matrix of observed field data and model simulation data
  // Note that this omits the prediction points for computational efficiency, as these are not required for the inverse problem of determining calibration parameters.
  // As such, an empty matrix of points is passed to the cal_pred_cov as x_pred, with n_pred set to 0.
  // An alternative, perhaps more elegant way of doing this would be instead to structure the covariance matrix according to parameters which depend on specific values of the calibration inputs,
  // i.e. the model ouputs, and those which do not, i.e. the field data and x_pred. A distinction wouldn't be made between the field data points and x_pred within the function, allowing this to 
  // be used to compute the covariance matrix both with and without predictions, provided the two sets of inputs were passed to the function correctly.
  // Consider implementing this later.
  sigma_z = cal_pred_cov(n, m, n+m, 0, p, q, xf, tf, xc, tc, rep_matrix(0,0,p), beta_eta, lambda_eta, beta_delta, lambda_delta, lambda_e);
  
// Specify priors here
  rho_eta[1:(p+q)] ~ beta(1.0, 0.3); // correlation parameter for emulator
  rho_delta[1:p] ~ beta(1.0, 0.3); // correlation parameter for discrepancy
  lambda_eta ~ gamma(10, 10); // precision parameter for emulator. gamma (shape, rate)
  lambda_delta ~ gamma(10, 0.3); // precision parameter for discrepancy
  lambda_e ~ gamma(10, 0.03); // precision parameter for observation error (1/standard deviation)

  L = cholesky_decompose(sigma_z); // cholesky decomposition of covariance
  
  // Tell Stan that y and eta are from a multivariate normal distribution with mean mu, and covariance sigma_z.
  // i.e. are field data and model outputs are assumed to be drawn from a Gaussian process with the specified covariance structure
  y_eta ~ multi_normal_cholesky(mu, L);
}

generated quantities {

  // This is the block of code in which the predictions are generated
  
  // declare variables
  matrix[N, N] sigma_z; // covariance matrix
  matrix[m+n, m+n] L; //  cholesky decomposition of covariance matrix
  vector[m+n] L_div_y;
  vector[m+n] K_div_y;
  vector[n_pred] f_mu; // posterior predictive mean at x_pred
  matrix[m+n, n_pred] v_pred;
  matrix[n_pred, n_pred] f_cov; // posterior predictive covariance for x_pred
  vector[n_pred] f_pred; // vector of prediction
  
  // Calcualte a joint covariance matrix of observed field data, model simulation points, and points at which predicitions are required
  sigma_z = cal_pred_cov(n, m, N, n_pred, p, q, xf, tf, xc, tc, x_pred, beta_eta, lambda_eta, beta_delta, lambda_delta, lambda_e);
  
  // Efficiently calculate the posterior predictive GP mean and covariance using the Cholesky decomposition of the covariance matrix
  // For expressions, see Chapter 2, "Gaussian Processes for Machine Learning", Rasmussen and Williams, MIT Press, 2006, ISBN 0-262-18253-X.
  // http://www.gaussianprocess.org/gpml/
  // Isolate portion of the covariance of field data and model output data
  L  = cholesky_decompose(sigma_z[1:(n+m),1:(n+m)]);
  L_div_y = mdivide_left_tri_low(L, y_eta);
  K_div_y = mdivide_right_tri_low(L_div_y', L)';
  // Calculate posterior predictive mean through matrix algebra
  f_mu = sigma_z[1:(n+m),(n+m+1):N]' * K_div_y;
  v_pred = mdivide_left_tri_low(L, sigma_z[1:n+m,(n+m+1):N]);
  // calcluate posterior predictive covariance through matrix algebra
  // note: a small nugget term is added to this to force the matrix to be positive-semi-definite when many predictions are required
  f_cov = sigma_z[(n+m+1):N,(n+m+1):N] - v_pred' * v_pred + diag_matrix(rep_vector(1e-8,n_pred));
  // sample predictions from a Gaussian process (discretised as multivariate normal) using posterior predictive mean and covariance
  f_pred = multi_normal_rng(f_mu, f_cov);
}

