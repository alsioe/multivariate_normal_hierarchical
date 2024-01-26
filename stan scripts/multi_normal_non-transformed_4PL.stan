//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  // Number of observations
  int<lower=0> n_y;
  // Number of subjects
  int<lower=0> n_subj;
  // Number of parameters
  int<lower=1> n_theta;
  // List of subjects per observation
  int<lower=1> subj_id[n_y];
  // x value each observation
  vector[n_y] x;
  // y value each observation
  vector[n_y] y;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  matrix[1, n_subj] alpha_by_subj_raw;
  matrix[1, n_subj] beta_by_subj_raw;
  matrix[1, n_subj] gamma_by_subj_raw;
  matrix[1, n_subj] delta_by_subj_raw;
  
  vector[1] alpha_mean;
  vector<lower = 0> [1] beta_mean;
  vector[1] gamma_mean;
  vector[1] delta_mean;

  vector<lower=0>[1] sd_alpha;
  vector<lower=0>[1] sd_beta;
  vector<lower=0>[1] sd_gamma;
  vector<lower=0>[1] sd_delta;

  // Standard deviation on noise
  real<lower=0> sd_y;
  
  // A single correlation matrix is assumed
  corr_matrix[n_theta] Omega;
}

transformed parameters {
  // We will place all parameters into a parameter object, theta
  vector[n_theta] theta_means;
  vector[n_theta] sd_theta;
  matrix[n_theta, n_subj] theta_by_subj_raw;
  
  // Transformed alphas and betas
  matrix[n_theta, n_subj] theta_by_subj;
  matrix[1, n_subj] alpha_by_subj;
  matrix<lower = 0>[1, n_subj] beta_by_subj;
  matrix[1, n_subj] gamma_by_subj;
  matrix[1, n_subj] delta_by_subj;
 
  // Covariance matrix
  cov_matrix[n_theta] Sigma;
  
  // Cholesky decomposition of the Sigma
  cholesky_factor_cov[n_theta] L;
  
  // Construct the combined parameters theta
  theta_means = append_row(alpha_mean,
                           append_row(beta_mean,
                                      append_row(gamma_mean,
                                                 delta_mean
                                                 )
                                      )
                           );
                           
  sd_theta = append_row(sd_alpha,
                        append_row(sd_beta,
                                   append_row(sd_gamma,
                                              sd_delta
                                              )
                                    )
                        );
  
  theta_by_subj_raw = append_row(alpha_by_subj_raw,
                                 append_row(beta_by_subj_raw,
                                            append_row(gamma_by_subj_raw,
                                                       delta_by_subj_raw
                                                       )
                                            )
                                );
  
  // Correlation (Omega) to covariance (Sigma)
  Sigma = quad_form_diag(Omega, sd_theta);
    
  // Cholesky factor from the covariance matrix
  L = cholesky_decompose(Sigma);
    
  // Matrix multiplication reduces to vectors
  for (s in 1:n_subj) {
    theta_by_subj[, s] = theta_means +
                         L * theta_by_subj_raw[, s];
                         }

  // Extract alpha and beta by subject
  alpha_by_subj = to_matrix(theta_by_subj[1, ]);
  beta_by_subj = to_matrix(theta_by_subj[2, ]);
  gamma_by_subj = to_matrix(theta_by_subj[3, ]);
  delta_by_subj = to_matrix(theta_by_subj[4, ]);
  
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  
  alpha_mean ~ normal(100, 10);
  beta_mean ~ normal(0, 10);
  gamma_mean ~ normal(0, 10);
  delta_mean ~ normal(0, 10);
  
  sd_alpha ~ weibull(2, 10);
  sd_beta ~ weibull(2, 1);
  sd_gamma ~ weibull(2, 10);
  sd_delta ~ weibull(2, 10);
  
  to_vector(alpha_by_subj_raw) ~ std_normal();
  to_vector(beta_by_subj_raw) ~ std_normal();
  to_vector(gamma_by_subj_raw) ~ std_normal();
  to_vector(delta_by_subj_raw) ~ std_normal();
  
  // Uniform prior on the correlation matrix (eta = 1)
  Omega ~ lkj_corr(1);
  
  sd_y ~ weibull(2, 2);
    
  for (t in 1:n_y) {

	y[t] ~ normal(delta_by_subj[1, subj_id[t]] +
                  ( alpha_by_subj[1, subj_id[t]] - delta_by_subj[1, subj_id[t]] ) /
                  ( 1 + exp(- beta_by_subj[1, subj_id[t]] * (x[t] - gamma_by_subj[1, subj_id[t]])) ),
                  sd_y);
                  }
}

