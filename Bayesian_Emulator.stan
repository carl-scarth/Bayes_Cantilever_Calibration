// Code for fitting a Gaussian process emulator using Bayesian inferrence, and making subsequent predictions
// Modified from: https://github.com/adChong/bc-stan, [1]
// prediction code modified from: 
// https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#21_Simulating_From_A_Gaussian_Process [2]
// The main modification is to strip out the experimental data from [1], leaving only the emulator.
// efficient posterior predictions are made using the closed-form Gaussian process expressions, with code taken from [2].
// For expressions, see Chapter 2, "Gaussian Processes for Machine Learning", Rasmussen and Williams, MIT Press, 2006, ISBN 0-262-18253-X.
// http://www.gaussianprocess.org/gpml/

functions {
  // Define a function for evaluating a square-exponential covariance matrix with non-isotropic correlation length i.e. there 
  // is a different correlation length for each input (this is necessary as the in-built Stan SE covariance is isotropic)
  matrix ARD_SE_cov(matrix x, real lambda, row_vector beta, real delta) {
    int N = rows(x);    // number of training data points
    int d = cols(x);    // number of inputs
    matrix[N, N] K;     // Covariance matrix
    {
    // Main body of the function
    // Declare variables
    row_vector[d] dx;   // variable used to determine difference between inputs
    K = diag_matrix(rep_vector((1/lambda), N));
    for (i in 1:N) {
      for (j in (i+1):N) {
        dx = x[i] - x[j];
        K[i, j] = (beta .* dx) * dx';
        K[i, j] = exp(-K[i, j])/lambda;
        K[j, i] = K[i, j];
      }
    }
    // add nugget term to diagonals
    K = K + diag_matrix(rep_vector(delta,N));
    }
    return K;
  }
  
  // Define a function for evaluating a square-exponential covariance matrix with non-isotropic correlation length as above,
  // but for a non-symmetric covariance matrix. This is used when evaulating the covariance between two different sets of points.
  matrix ARD_SE_cov_non_sym(matrix x1, matrix x2, real lambda, row_vector beta) {
    int N1 = rows(x1);  // number of training data points
    int N2 = rows(x2);  // number of points at which predictions are required
    int d = cols(x1);   // number of inputs
    matrix[N1, N2] K;   // Declare covariance matrix
    {
    // Main body of the function
    // Declare variables
    row_vector[d] dx;   // variable used to determine difference between inputs
    K = rep_matrix(0,N1,N2);
    for (i in 1:N1) {
      for (j in 1:N2) {
        dx = x1[i] - x2[j];
        K[i, j] = (beta .* dx) * dx';
        K[i, j] = exp(-K[i, j])/lambda;
      }
    }
    }
    return K;
  }
  
  // Define a function for analytical expressions for GP posterior predictions for a zero-mean prior, and squared-exponential covariance
  vector gp_pred_rng(matrix x_t, vector y, matrix x, real lambda, row_vector beta, real delta) {
    int N1 = rows(y);   // number of training data points
    int N2 = rows(x_t); // number of test points at which predictions are required
    int d = cols(x);    // number of inputS
    vector[N2] f_pred;  // declare function output, a vector of predictions
    {
    // Main body of the function
    // Declare variables
    matrix[N1, N1] K;       // Covariance matrix of training data points with themselves
    matrix[N1, N1] L_K;     // Cholesky decomposition of covariance matrix
    vector[N1] L_K_div_y;
    vector[N1] K_div_y;
    matrix[N1,N2] k_x_xt;   // Covariance matrix of training data points with test data points
    vector[N2] f_mu;        // Posterior predictive GP mean
    matrix[N1, N2] v_pred;
    matrix[N2, N2] C_xt_xt; // Covariance matrix of test data points with themselves
    matrix[N2, N2] f_cov;   // Posterior predictive GP covariance
    // Define covariance matrix combining training and test data
    K = ARD_SE_cov(x, lambda, beta, 0);
    // Perform Cholesky decomposition of covariance matrix for subsequent efficient matrix algebra
    L_K  = cholesky_decompose(K);
    // Define covariance matrix of training data with test data. Note that this matrix is not symmetric
    k_x_xt = ARD_SE_cov_non_sym(x, x_t, lambda, beta);
    // Determine covariance of test points in x_t with themselves
    // note: a small nugget term, delta, is added to ensure covariance is positive-semi-definite when many predictions are required
    C_xt_xt = ARD_SE_cov(x_t, lambda, beta, delta);
    // Efficient code for calculating emulator predictive mean, k_x_xt^T * K^-1 * y
    L_K_div_y = mdivide_left_tri_low(L_K, y);
    K_div_y = mdivide_right_tri_low(L_K_div_y', L_K)'; // K^-1 * y
    // Evaluate posterior predictive mean
    f_mu = (k_x_xt' * K_div_y); // posterior mean
    // Now cacluate posterior predictive covariance, C_xt_xt - k_x_xt^T * K^-1 * k_x_xt
    v_pred = mdivide_left_tri_low(L_K, k_x_xt);
    // calcluate posterior predictive covariance
    f_cov = C_xt_xt - v_pred' * v_pred;
    // Sample from Gaussian process
    f_pred = multi_normal_rng(f_mu, f_cov);
    }
    return f_pred;
  }
}

data {
  // Note: In this code I have separated inputs into "controlled" and "uncontrolled" inputs, as would be the case in calibration
  // This is not strictly necessary for training an emulator, which does not care about the type of inputs. I have retained this
  // distinction to minimise the amount of changes required to pre-processing when switching between calibration and emulator codes
  int<lower=0> m;           // number of computer simulation
  int<lower=0> p;           // number of "controlled" inputs, x
  int<lower=0> q;           // number of "uncontrolled" inputs, t
  int<lower=1> n_pred;      // number of points at which predictions are required
  vector[m] eta;            // output of computer simulations
  matrix[m, p] xc;          // training data matrix of "controlled" inputs
  matrix[m, q] tc;          // training data matrix of "uncontrolled" inputs
  matrix[n_pred, p] x_pred; // "controlled" input values at which predicitons are required
  matrix[n_pred, q] t_pred; // "uncontrolled" input values at which predictions are required
}

transformed data {
  vector[m] mu; // prior mean vector
  matrix[m, p+q] xt; // joint matrix of "controlled" and "uncontrolled" training data points
  matrix[n_pred, p+q] xt_pred; // joint matrix of "controlled" and "uncontrolled" test data points
  mu = rep_vector(0, m); // set prior mean vector to zero
  // combine "controlled" and "uncontrolled" inputs into a joint matrix (both training and test data)
  xt = append_col(xc, tc);
  xt_pred = append_col(x_pred, t_pred);
}

parameters {
  // rho_eta: reparameterisation of correlation length, beta_eta
  // lambda_eta: precision parameter for eta
  row_vector<lower=0,upper=1>[p+q] rho_eta;
  real<lower=0> lambda_eta; 
}

transformed parameters {
  // cacluate correlation length from transformed parameter rho_eta
  row_vector[p+q] beta_eta;
  beta_eta = -4.0 * log(rho_eta);
}

model {
  // declare variables
  matrix[m, m] sigma_eta;   // Covariance matrix
  matrix[m, m] L;           // cholesky decomposition of covariance matrix 
  row_vector[p+q] temp_eta; 
  
  // Calculate covariance matrix of training data points
  sigma_eta = ARD_SE_cov(xt, lambda_eta, beta_eta, 0);
  
  // Specify priors here
  rho_eta[1:(p+q)] ~ beta(1.0, 0.3);    // Correlation parameter
  lambda_eta ~ gamma(10, 10);           // Precision parameter, gamma (shape, rate)
  L = cholesky_decompose(sigma_eta); // cholesky decomposition
  // Tell Stan that eta is from a multivariate normal distribution with mean mu, and covariance sigma_eta.
  // i.e. model outputs are assumed to be drawn from a Gaussian process with the specified covariance structure
  eta ~ multi_normal_cholesky(mu, L);
}

generated quantities {
  // use custom function to sample from GP posterior predictive distribution using the inferred emulator hyperparameters
  vector[n_pred] f_pred;
  f_pred = gp_pred_rng(xt_pred, eta, xt, lambda_eta, beta_eta, 1e-10);
}

