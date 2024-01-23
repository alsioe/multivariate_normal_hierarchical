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
  vector[n_subj] alpha_by_subj;
  vector[n_subj] beta_by_subj;
  
  real alpha_mean;
  real beta_mean;
  
  real<lower=0> sd_alpha;
  real<lower=0> sd_beta;
  
  // Standard deviation on noise
  real<lower=0> sd_y;
  
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  
  alpha_mean ~ normal(0, 10);
  beta_mean ~ normal(0, 10);
  
  sd_alpha ~ weibull(2, 10);
  sd_beta ~ weibull(2, 1);
  
  alpha_by_subj ~ normal(alpha_mean, sd_alpha);
  beta_by_subj ~ normal(beta_mean, sd_beta);
  
  sd_y ~ weibull(2, 2);
    
  y ~ normal(alpha_by_subj[subj_id] + beta_by_subj[subj_id] .* x,
             sd_y);
}

