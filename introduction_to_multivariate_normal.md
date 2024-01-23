---
title: "Multivariate normal strategy"
author: "Johan Alsiö"
date: "2024-01-22"
output:
    html_document:
        keep_md: true
---



## Multivariate normal hierarchical models

A number of different data sets will be explored.
1. One population with a normal distribution, different-sized samples

Modelling data.

We will randomise some data for our first example using the rnorm() function,
which draws random samples from a normal distribution with 'location' (mean) and
'scale' (sd = standard deviation). Note that we need many draws to be able to 
observe the normal distribution visually.


```r
data <- rbind(data.frame(group = 'V', value = rnorm(5, 0, 1)),
              data.frame(group = 'X', value = rnorm(10, 0, 1)),
              data.frame(group = 'L', value = rnorm(50, 0, 1)),
              data.frame(group = 'C', value = rnorm(100, 0, 1))
              )
    
data$group <- factor(x = data$group,
                    levels = c('V', 'X', 'L', 'C')
                    )

# Plot as histograms
ggplot(data = data,
       mapping = aes(x = value)
       ) +
    facet_wrap(~ group,
               scales = 'free') +
    geom_histogram() +
    theme_bw()
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](introduction_to_multivariate_normal_files/figure-html/normal distribution-1.png)<!-- -->

```r
# Plotting as densities, with the true probability density in red
ggplot(data = data,
       mapping = aes(x = value)
       ) +
    facet_wrap(~ group,
               scales = 'free') +
    geom_density() +
    geom_function(fun = dnorm,
                  colour = "red",
                  linetype = 'dashed') +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/normal distribution-2.png)<!-- -->


```r
# We can ask Stan to find the parameters of any or all of those
normal_data_for_stan <- list(n_y = dim(data)[1],
                             n_groups = length(unique(data$group)),
                             group = match(data$group, unique(data$group)),
                             y = data$value)

fit_normal <- rstan::stan(file = 'stan scripts/normal_distribution.stan',
                          model_name = 'normal distribution',
                          data = normal_data_for_stan,
                          chains = 4,
                          cores = 4,
                          iter = 4000,
                          warmup = 2000,
                          control = list(adapt_delta = 0.99)
                          ) 

# Inspect the trace plots - we want them to look like 'hairy caterpillars''
# https://druedin.com/2016/12/26/that-hairy-caterpillar/
rstan::traceplot(fit_normal)
```

![](introduction_to_multivariate_normal_files/figure-html/Stan model for simple, normal distributions-1.png)<!-- -->

```r
# Create summary object and print out the overall summary (a.k.a. summary$summary)
summary <- rstan::summary(fit_normal,
                     probs = c(0.025, 0.5, 0.975)
                     )

# We want all the Rhats to be less than 1.01, and we need n_eff to be >100 but
# ideally in the thousands
round(summary$summary, 2)
```

```
##            mean se_mean   sd   2.5%    50%  97.5%   n_eff Rhat
## mu[1]      0.10    0.02 1.35  -2.71   0.10   2.94 3438.11    1
## mu[2]      0.05    0.01 0.40  -0.73   0.05   0.87 4394.23    1
## mu[3]     -0.20    0.00 0.13  -0.45  -0.19   0.05 7119.93    1
## mu[4]      0.02    0.00 0.11  -0.20   0.02   0.24 7595.87    1
## sigma[1]   2.78    0.03 1.67   1.08   2.30   7.44 2990.90    1
## sigma[2]   1.19    0.01 0.37   0.70   1.12   2.11 4571.22    1
## sigma[3]   0.89    0.00 0.09   0.74   0.89   1.10 7654.23    1
## sigma[4]   1.08    0.00 0.08   0.95   1.08   1.25 7247.97    1
## lp__     -84.71    0.04 2.17 -89.86 -84.34 -81.52 2870.26    1
```

```r
# Extract the posterior distributions from the fit
par <- rstan::extract(fit_normal)

# Save all samples from the posterior into a data frame
df <- data.frame(
    iter = length(par$mu[, 1]),
    V_mu = par$mu[, 1],
    X_mu = par$mu[, 2],
    L_mu = par$mu[, 3],
    C_mu = par$mu[, 4],
    V_sigma = par$sigma[, 1],
    X_sigma = par$sigma[, 2],
    L_sigma = par$sigma[, 3],
    C_sigma = par$sigma[, 4]
    )

# Re-format for easy graphing
df_long <- pivot_longer(data = df,
                        cols = !iter,
                        names_to = 'parameter',
                        values_to = 'sample')

par_names <- c('V_mu', 'X_mu', 'L_mu', 'C_mu',
               'V_sigma', 'X_sigma', 'L_sigma', 'C_sigma')

# Make the parameter column a factor for easier data handling and graphing
df_long$parameter <- factor(x = df_long$parameter,
                            levels = par_names
                            )

# Plot the probability densities with vertical lines (geom_vline) at 'real' value
ggplot(data = df_long,
       mapping = aes(x = sample)
       ) +
    facet_wrap(~ parameter,
               scales = 'free',
               ncol = 4) +
    geom_density() +
    geom_vline(data = data.frame(parameter = factor(par_names, levels = par_names),
                          xintercept = c(0, 0, 0, 0, 1, 1, 1, 1)
                          ),
               aes(xintercept = xintercept),
               colour = 'red',
               linetype = 'dashed') +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/Stan model for simple, normal distributions-2.png)<!-- -->

If we sample from two different values with appropriate distributions, we can use
the values to simulate data from a linear relationship, y = alpha + beta*x


```r
# samples to draw from each distribution
N <- 10

true_linear <- data.frame(
        subj_id = 1:N,
        alpha = rnorm(N, 10, 2),
        beta = rnorm(N, 1, 0.5)
        )

true_linear_long <- true_linear %>%
    pivot_longer(cols = !subj_id,
                 names_to = 'parameter',
                 values_to = 'value')

true_linear_long$parameter <- factor(x = true_linear_long$parameter,
                              levels = c('alpha', 'beta')
                              )

# Set up a function to get from the parameters to the simulated data
linear <- function(alpha, beta, x) {
    y = alpha + beta * x

    return(y)
}

# Plot the histograms
ggplot(data = true_linear_long,
       mapping = aes(x = value)
       ) +
    facet_wrap(~ parameter,
               scales = 'free') +
    geom_histogram() +
    theme_bw()
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

![](introduction_to_multivariate_normal_files/figure-html/many normal distributions-1.png)<!-- -->

```r
# Set up a palette
my_palette <- viridis::viridis(n = N)

# Set up canvas
plot <- ggplot() +
        xlim(-10, 10) +
        theme_bw()

# Add the individual curves    
for (n in 1:N) {
    plot <- plot +
    geom_function(fun = linear,
                  args = list(alpha = true_linear$alpha[n],
                              beta = true_linear$beta[n]
                              ),
                  colour = my_palette[n]
                  )        
}

plot
```

![](introduction_to_multivariate_normal_files/figure-html/many normal distributions-2.png)<!-- -->

Any true data will also come with measurement noise etc., so we will add a extra layer. 


```r
# We now know how to draw perfect data - but we want noisy data!
# Let's add some noise and collect data in duplicates at various x values

n_rep <- 2
x_series <- seq(from = -10, to = 10, by = 5)

# Set up a data frame with the data for the linear association
normal_data <- data.frame(
    subj_id = gl(n = N, k = n_rep * length(x_series)),
    rep = rep(x = 1:n_rep,
              times = N * length(x_series)),
    x = rep(x = sort(rep(x_series, n_rep)),
            times = N),
    y = NA
)

# In this (simplistic) case we will have a common noise level on all data
sd_y <- 10

normal_data$y <- linear(alpha = true_linear$alpha[normal_data$subj_id],
                        beta = true_linear$beta[normal_data$subj_id],
                        x = normal_data$x) +
                 rnorm(n = length(normal_data$y),
                       mean = 0,
                       sd = sd_y)

# Example plots (the filter makes sure we only plot up to 4 subjects)
ggplot(data = filter(normal_data, subj_id %in% c(1:4)),
       mapping = aes(x = x,
                     y = y,
                     colour = subj_id)) +
    geom_jitter(width = 0.1) +
    geom_smooth(method = 'lm') +
    theme_bw()
```

```
## `geom_smooth()` using formula = 'y ~ x'
```

![](introduction_to_multivariate_normal_files/figure-html/noisy data from two normal distributions-1.png)<!-- -->


```r
# ... and we can now ask Stan to try to estimate the starting parameters!
# 

# We can ask Stan to find the parameters of any or all of those
linear_data_for_stan <- list(n_y = dim(normal_data)[1],
                             n_subj = length(unique(normal_data$subj_id)),
                             subj_id = as.integer(normal_data$subj_id),
                             x = normal_data$x,
                             y = normal_data$y)

# Note to self: the standard deviations are not informed by the data so should
# be removed from this, the simplest model!

fit_linear <- rstan::stan(file = 'stan scripts/linear.stan',
                          model_name = 'linear',
                          data = linear_data_for_stan,
                          chains = 4,
                          cores = 4,
                          iter = 4000,
                          warmup = 2000,
                          control = list(adapt_delta = 0.99)
                          ) 

# Inspect the trace plots - we want them to look like 'hairy caterpillars''
# https://druedin.com/2016/12/26/that-hairy-caterpillar/
rstan::traceplot(fit_linear)
```

```
## 'pars' not specified. Showing first 10 parameters by default.
```

![](introduction_to_multivariate_normal_files/figure-html/stan to fit linear model-1.png)<!-- -->

```r
# Create summary object and print out the overall summary (a.k.a. summary$summary)
summary <- rstan::summary(fit_linear,
                     probs = c(0.025, 0.5, 0.975)
                     )

# We want all the Rhats to be less than 1.01, and we need n_eff to be >100 but
# ideally in the thousands
round(summary$summary, 2)
```

```
##                      mean se_mean   sd    2.5%     50%   97.5%    n_eff Rhat
## alpha_by_subj[1]     9.07    0.02 2.54    4.10    9.08   14.06 11645.23    1
## alpha_by_subj[2]     8.56    0.02 2.61    3.45    8.56   13.62 12511.17    1
## alpha_by_subj[3]    13.31    0.02 2.50    8.47   13.30   18.30 13094.86    1
## alpha_by_subj[4]    11.49    0.02 2.56    6.48   11.48   16.58 13452.28    1
## alpha_by_subj[5]    15.94    0.02 2.56   10.92   15.93   20.99 13401.78    1
## alpha_by_subj[6]     8.40    0.02 2.53    3.44    8.44   13.35 11704.26    1
## alpha_by_subj[7]     6.67    0.02 2.57    1.54    6.68   11.63 11424.56    1
## alpha_by_subj[8]    14.02    0.02 2.55    9.03   13.97   18.97 12956.60    1
## alpha_by_subj[9]    12.76    0.02 2.60    7.66   12.76   17.81 13680.89    1
## alpha_by_subj[10]   10.78    0.02 2.55    5.78   10.78   15.83 12800.30    1
## beta_by_subj[1]      0.67    0.00 0.37   -0.06    0.67    1.41 12011.64    1
## beta_by_subj[2]      0.69    0.00 0.37   -0.04    0.69    1.40 11591.14    1
## beta_by_subj[3]      0.83    0.00 0.38    0.09    0.83    1.57 11867.04    1
## beta_by_subj[4]      1.62    0.00 0.38    0.89    1.62    2.35 12780.74    1
## beta_by_subj[5]      0.74    0.00 0.37    0.02    0.75    1.47 11856.25    1
## beta_by_subj[6]      0.75    0.00 0.38    0.00    0.75    1.48 11016.40    1
## beta_by_subj[7]      1.02    0.00 0.37    0.31    1.02    1.76 12069.95    1
## beta_by_subj[8]      1.48    0.00 0.38    0.75    1.49    2.22 11862.06    1
## beta_by_subj[9]      1.09    0.00 0.37    0.36    1.09    1.82 10579.25    1
## beta_by_subj[10]     0.87    0.00 0.36    0.16    0.87    1.57 11759.40    1
## sd_alpha             8.90    0.04 4.62    1.53    8.34   19.19 11343.62    1
## sd_beta              0.88    0.00 0.47    0.15    0.83    1.94 11435.00    1
## sd_y                 8.30    0.01 0.49    7.41    8.29    9.32  8598.88    1
## lp__              -296.42    0.06 3.63 -304.64 -296.06 -290.35  3276.09    1
```

```r
# Extract the posterior distributions from the fit
par <- rstan::extract(fit_linear)

par_names <- c('alpha_by_subj',
               'beta_by_subj',
               'sd_alpha',
               'sd_beta',
               'sd_y')

for (s in 1:N) {
    if (s == 1) {
        df <- data.frame(
            subj_id = vector(),
            iter = vector(),
            alpha_by_subj = vector(),
            beta_by_subj = vector()
        )
    }
    
    df <- rbind(df,
                data.frame(
                    subj_id = rep(s, length(par$alpha_by_subj[, s])),
                    iter = 1:length(par$alpha_by_subj[, s]),
                    alpha_by_subj = par$alpha_by_subj[, s],
                    beta_by_subj = par$beta_by_subj[, s]
                    )
                )
}

df_long <- pivot_longer(data = df,
                        cols = !c(subj_id, iter),
                        names_to = 'parameter',
                        values_to = 'sample')

df_long <- rbind(df_long,
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_alpha',
                    sample = par$sd_alpha
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_beta',
                    sample = par$sd_beta
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_y',
                    sample = par$sd_y
                    )
                 )
    
par_names <- unique(df_long$parameter)

# Make the parameter column a factor for easier data handling and graphing
df_long$parameter <- factor(x = df_long$parameter,
                            levels = par_names
                            )

# Plot the probability densities with vertical lines (geom_vline) at 'real' value
ggplot(data = df_long,
       mapping = aes(x = sample,
                     colour = as.factor(subj_id))
       ) +
    facet_wrap(~ parameter,
               scales = 'free',
               ncol = 4) +
    geom_density() +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/stan to fit linear model-2.png)<!-- -->

```r
# Quantile function with updated probs
# https://www.tidyverse.org/blog/2023/02/dplyr-1-1-0-pick-reframe-arrange/
quantile_df <- function(x, probs = c(0.025, 0.5, 0.975)) {
  tibble(
    value = quantile(x, probs, na.rm = TRUE),
    prob = probs
  )
}

# ALPHA PARAMETER
# 
# plot(x = true_linear$beta,
#      y = colMeans(par$beta_by_subj)
#      )

df_wide_quantiles <- df_long %>%
                        filter(parameter == 'alpha_by_subj') %>%
                        reframe(quantile_df(sample), .by = subj_id) %>%
                        pivot_wider(id_cols = c(subj_id),
                                    names_from = prob,
                                    values_from = value)

names(df_wide_quantiles) <- c('subj_id', 'low', 'median', 'high')

ggplot(data = df_wide_quantiles,
       mapping = aes(y = median,
                     ymin = low,
                     ymax = high,
                     colour = as.factor(subj_id))
       ) +
    geom_point(aes(x = true_linear$alpha,
                   y = true_linear$alpha)) +
    geom_errorbar(aes(x = true_linear$alpha),
                  width=.2) +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/stan to fit linear model-3.png)<!-- -->

```r
# BETA PARAMETER
df_wide_quantiles <- df_long %>%
                        filter(parameter == 'beta_by_subj') %>%
                        reframe(quantile_df(sample), .by = subj_id) %>%
                        pivot_wider(id_cols = c(subj_id),
                                    names_from = prob,
                                    values_from = value)

names(df_wide_quantiles) <- c('subj_id', 'low', 'median', 'high')

ggplot(data = df_wide_quantiles,
       mapping = aes(y = median,
                     ymin = low,
                     ymax = high,
                     colour = as.factor(subj_id))
       ) +
    geom_point(aes(x = true_linear$beta,
                   y = true_linear$beta)) +
    geom_errorbar(aes(x = true_linear$beta),
                  width=.1) +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/stan to fit linear model-4.png)<!-- -->

The estimates are very wide at this point, representing the fact that the measurement
noise (sd_y) is quite high, and we only have two measurements per x value. If we were
able to re-run the experiment, perhaps we would improve the measurements - or simply
try to collect more measurements per x value! But if we cannot easily re-do the
experiment, we can still improve the fit by making some additions to the model.

i) Assume that the individuals represent draws from a normally distributed population.
ii) Assume that the parameters come from a multivariate normal distribution


```r
# The updated stan file is using hierarchical priors on the subjects' parameters.
# alpha_by_subj ~ normal(alpha_mean, sd_alpha;
hierarchical_linear_fit <- rstan::stan(file = 'stan scripts/centered_hierarchical_linear.stan',
                                       model_name = 'hierarchical_linear',
                                       data = linear_data_for_stan,
                                       chains = 4,
                                       cores = 4,
                                       iter = 4000,
                                       warmup = 2000,
                                       control = list(adapt_delta = 0.99)
                                       ) 
```

```
## Warning: There were 7 divergent transitions after warmup. See
## https://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
## to find out why this is a problem and how to eliminate them.
```

```
## Warning: Examine the pairs() plot to diagnose sampling problems
```

```r
# There are a number of error messages about divergent transitions - this is standard
# when running not-yet optimised hierarchical models and it is not a major concern
# until they start to ramp up in numbers in more complex models. We can sort them
# straight away though - these divergent transitions are caused by the
# correlations between the parameters-by-subj and the group mean.
# A simple trick is to used the 'non-centered parameterisation' which
# treats each subject's value as the sum of the mean and some 'raw' subject effect,
# which is the difference between the subject's value and the mean, measured in
# standard deviations.
# alpha_by_subj = alpha_mean + alpha_by_subj_raw * sd_alpha;
# We then put a 'standard normal' prior on the raw subject value
# alpha_by_subj_raw ~ normal(0, 3);
# (Note that the function std_normal() is identifical to normal(0, 3))

hierarchical_linear_fit <- rstan::stan(file = 'stan scripts/hierarchical_linear.stan',
                                       model_name = 'hierarchical_linear',
                                       data = linear_data_for_stan,
                                       chains = 4,
                                       cores = 4,
                                       iter = 4000,
                                       warmup = 2000,
                                       control = list(adapt_delta = 0.99)
                                       )

# Hopefully no divergent transitions! The alpha_by_subj_raw does NOT interact
# in the same way with the overall mean, and the sampler runs smoothly.

# Inspect the trace plots - we want them to look like 'hairy caterpillars''
# https://druedin.com/2016/12/26/that-hairy-caterpillar/
rstan::traceplot(hierarchical_linear_fit)
```

```
## 'pars' not specified. Showing first 10 parameters by default.
```

![](introduction_to_multivariate_normal_files/figure-html/introducing hierarchical modelling-1.png)<!-- -->

```r
# Create summary object and print out the overall summary (a.k.a. summary$summary)
summary <- rstan::summary(hierarchical_linear_fit,
                     probs = c(0.025, 0.5, 0.975)
                     )

# We want all the Rhats to be less than 1.01, and we need n_eff to be >100 but
# ideally in the thousands
round(summary$summary, 2)
```

```
##                          mean se_mean   sd    2.5%     50%   97.5%    n_eff
## alpha_by_subj_raw[1]    -0.34    0.01 0.76   -1.86   -0.34    1.15  7916.53
## alpha_by_subj_raw[2]    -0.44    0.01 0.75   -1.93   -0.44    1.06  7613.61
## alpha_by_subj_raw[3]     0.46    0.01 0.75   -1.04    0.46    1.93  9003.39
## alpha_by_subj_raw[4]     0.10    0.01 0.74   -1.36    0.10    1.58  8243.88
## alpha_by_subj_raw[5]     0.92    0.01 0.76   -0.62    0.93    2.42  7704.86
## alpha_by_subj_raw[6]    -0.46    0.01 0.74   -1.92   -0.46    1.03  8262.04
## alpha_by_subj_raw[7]    -0.79    0.01 0.77   -2.30   -0.78    0.73  8513.61
## alpha_by_subj_raw[8]     0.58    0.01 0.74   -0.91    0.59    2.04  8274.19
## alpha_by_subj_raw[9]     0.34    0.01 0.74   -1.16    0.35    1.78  7935.65
## alpha_by_subj_raw[10]   -0.03    0.01 0.74   -1.50   -0.02    1.44  7862.04
## beta_by_subj_raw[1]     -0.35    0.01 0.84   -2.02   -0.36    1.31  9265.50
## beta_by_subj_raw[2]     -0.32    0.01 0.84   -2.01   -0.32    1.37  8822.07
## beta_by_subj_raw[3]     -0.17    0.01 0.84   -1.83   -0.17    1.49  8128.60
## beta_by_subj_raw[4]      0.73    0.01 0.84   -0.96    0.76    2.39  6866.07
## beta_by_subj_raw[5]     -0.27    0.01 0.82   -1.87   -0.28    1.35  7595.99
## beta_by_subj_raw[6]     -0.27    0.01 0.85   -1.95   -0.27    1.44  9259.27
## beta_by_subj_raw[7]      0.05    0.01 0.83   -1.62    0.06    1.72  8814.09
## beta_by_subj_raw[8]      0.58    0.01 0.84   -1.15    0.59    2.19  8186.42
## beta_by_subj_raw[9]      0.12    0.01 0.84   -1.57    0.13    1.75  9772.39
## beta_by_subj_raw[10]    -0.11    0.01 0.84   -1.74   -0.12    1.56  8883.37
## alpha_mean              11.66    0.02 1.28    8.95   11.69   14.14  4041.09
## beta_mean                0.98    0.00 0.15    0.67    0.98    1.29  5413.53
## sd_alpha                 2.89    0.03 1.44    0.67    2.69    6.30  2490.55
## sd_beta                  0.28    0.00 0.16    0.05    0.26    0.65  3569.75
## sd_y                     8.21    0.01 0.47    7.34    8.20    9.19  8651.00
## alpha_by_subj[1]        10.71    0.02 2.00    6.51   10.78   14.47  9668.03
## alpha_by_subj[2]        10.42    0.02 2.02    6.18   10.54   14.06  9339.81
## alpha_by_subj[3]        13.01    0.02 1.99    9.33   12.90   17.24 10376.88
## alpha_by_subj[4]        12.00    0.02 1.91    8.29   11.99   15.85 11724.82
## alpha_by_subj[5]        14.36    0.03 2.22   10.45   14.21   19.08  6438.85
## alpha_by_subj[6]        10.35    0.02 2.03    6.10   10.48   14.04  8413.66
## alpha_by_subj[7]         9.39    0.03 2.23    4.61    9.52   13.29  6470.01
## alpha_by_subj[8]        13.35    0.02 2.01    9.71   13.25   17.56  8727.55
## alpha_by_subj[9]        12.66    0.02 1.97    8.88   12.59   16.71 10642.39
## alpha_by_subj[10]       11.61    0.02 1.94    7.66   11.62   15.37 11390.93
## beta_by_subj[1]          0.87    0.00 0.24    0.33    0.89    1.32 10882.26
## beta_by_subj[2]          0.88    0.00 0.24    0.38    0.89    1.33  9974.72
## beta_by_subj[3]          0.93    0.00 0.24    0.42    0.93    1.39  9761.23
## beta_by_subj[4]          1.20    0.00 0.27    0.75    1.17    1.80  6777.38
## beta_by_subj[5]          0.89    0.00 0.24    0.37    0.91    1.36 10733.55
## beta_by_subj[6]          0.90    0.00 0.24    0.37    0.91    1.35 11692.62
## beta_by_subj[7]          0.99    0.00 0.24    0.51    0.99    1.48 11730.02
## beta_by_subj[8]          1.15    0.00 0.26    0.70    1.13    1.74  7479.60
## beta_by_subj[9]          1.01    0.00 0.24    0.56    1.01    1.50 11058.17
## beta_by_subj[10]         0.95    0.00 0.24    0.46    0.95    1.41 12049.73
## lp__                  -299.30    0.11 4.58 -308.88 -299.06 -291.10  1898.50
##                       Rhat
## alpha_by_subj_raw[1]     1
## alpha_by_subj_raw[2]     1
## alpha_by_subj_raw[3]     1
## alpha_by_subj_raw[4]     1
## alpha_by_subj_raw[5]     1
## alpha_by_subj_raw[6]     1
## alpha_by_subj_raw[7]     1
## alpha_by_subj_raw[8]     1
## alpha_by_subj_raw[9]     1
## alpha_by_subj_raw[10]    1
## beta_by_subj_raw[1]      1
## beta_by_subj_raw[2]      1
## beta_by_subj_raw[3]      1
## beta_by_subj_raw[4]      1
## beta_by_subj_raw[5]      1
## beta_by_subj_raw[6]      1
## beta_by_subj_raw[7]      1
## beta_by_subj_raw[8]      1
## beta_by_subj_raw[9]      1
## beta_by_subj_raw[10]     1
## alpha_mean               1
## beta_mean                1
## sd_alpha                 1
## sd_beta                  1
## sd_y                     1
## alpha_by_subj[1]         1
## alpha_by_subj[2]         1
## alpha_by_subj[3]         1
## alpha_by_subj[4]         1
## alpha_by_subj[5]         1
## alpha_by_subj[6]         1
## alpha_by_subj[7]         1
## alpha_by_subj[8]         1
## alpha_by_subj[9]         1
## alpha_by_subj[10]        1
## beta_by_subj[1]          1
## beta_by_subj[2]          1
## beta_by_subj[3]          1
## beta_by_subj[4]          1
## beta_by_subj[5]          1
## beta_by_subj[6]          1
## beta_by_subj[7]          1
## beta_by_subj[8]          1
## beta_by_subj[9]          1
## beta_by_subj[10]         1
## lp__                     1
```

```r
# Extract the posterior distributions from the fit
par <- rstan::extract(hierarchical_linear_fit)

par_names <- c('alpha_mean',
               'beta_mean',
               'alpha_by_subj',
               'beta_by_subj',
               'sd_alpha',
               'sd_beta',
               'sd_y')

for (s in 1:N) {
    if (s == 1) {
        df <- data.frame(
            subj_id = vector(),
            iter = vector(),
            alpha_by_subj = vector(),
            beta_by_subj = vector()
        )
    }
    
    df <- rbind(df,
                data.frame(
                    subj_id = rep(s, length(par$alpha_by_subj[, s])),
                    iter = 1:length(par$alpha_by_subj[, s]),
                    alpha_by_subj = par$alpha_by_subj[, s],
                    beta_by_subj = par$beta_by_subj[, s]
                    )
                )
}

df_long <- pivot_longer(data = df,
                        cols = !c(subj_id, iter),
                        names_to = 'parameter',
                        values_to = 'sample')

df_long <- rbind(df_long,
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'alpha_mean',
                    sample = par$alpha_mean
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'beta_mean',
                    sample = par$beta_mean
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_alpha',
                    sample = par$sd_alpha
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_beta',
                    sample = par$sd_beta
                    ),
                 data.frame(
                    subj_id = rep(0, length(par$sd_y)),
                    iter = 1:length(length(par$sd_y)),
                    parameter = 'sd_y',
                    sample = par$sd_y
                    )
                 )
    
par_names <- unique(df_long$parameter)

# Make the parameter column a factor for easier data handling and graphing
df_long$parameter <- factor(x = df_long$parameter,
                            levels = par_names
                            )

# Plot the probability densities with vertical lines (geom_vline) at 'real' value
ggplot(data = df_long,
       mapping = aes(x = sample,
                     colour = as.factor(subj_id))
       ) +
    facet_wrap(~ parameter,
               scales = 'free',
               ncol = 4) +
    geom_density() +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/introducing hierarchical modelling-2.png)<!-- -->

```r
# Quantile function with updated probs
# https://www.tidyverse.org/blog/2023/02/dplyr-1-1-0-pick-reframe-arrange/
quantile_df <- function(x, probs = c(0.025, 0.5, 0.975)) {
  tibble(
    value = quantile(x, probs, na.rm = TRUE),
    prob = probs
  )
}

# ALPHA PARAMETER
# 
# plot(x = true_linear$beta,
#      y = colMeans(par$beta_by_subj)
#      )

df_wide_quantiles <- df_long %>%
                        filter(parameter == 'alpha_by_subj') %>%
                        reframe(quantile_df(sample), .by = subj_id) %>%
                        pivot_wider(id_cols = c(subj_id),
                                    names_from = prob,
                                    values_from = value)

names(df_wide_quantiles) <- c('subj_id', 'low', 'median', 'high')

ggplot(data = df_wide_quantiles,
       mapping = aes(y = median,
                     ymin = low,
                     ymax = high,
                     colour = as.factor(subj_id))
       ) +
    geom_point(aes(x = true_linear$alpha,
                   y = true_linear$alpha)) +
    geom_errorbar(aes(x = true_linear$alpha),
                  width=.2) +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/introducing hierarchical modelling-3.png)<!-- -->

```r
# BETA PARAMETER
df_wide_quantiles <- df_long %>%
                        filter(parameter == 'beta_by_subj') %>%
                        reframe(quantile_df(sample), .by = subj_id) %>%
                        pivot_wider(id_cols = c(subj_id),
                                    names_from = prob,
                                    values_from = value)

names(df_wide_quantiles) <- c('subj_id', 'low', 'median', 'high')

ggplot(data = df_wide_quantiles,
       mapping = aes(y = median,
                     ymin = low,
                     ymax = high,
                     colour = as.factor(subj_id))
       ) +
    geom_point(aes(x = true_linear$beta,
                   y = true_linear$beta)) +
    geom_errorbar(aes(x = true_linear$beta),
                  width=.1) +
    theme_bw()
```

![](introduction_to_multivariate_normal_files/figure-html/introducing hierarchical modelling-4.png)<!-- -->

Introduce correlations between parameters


```r
# In the previous example, we can check visually whether the two parameters are related.
# 
# plot(x = true_parameters$alpha,
#      y = true_parameters$beta)

# In the real world, parameters can often be correlated. Imagine for instance
# that subjects with a high intercept have a lower slope parameter.
# We can simulate this using the MASS package and the function mvrnorm().
# What this function does is to create a multivariate normal distribution,
# which is simply a set of k parameters each of which is normally distributed
# and which share some covariance (correlations).

# The function mvrnorm() requires k means but and we then need to supply a
# k x k covariance matrix Sigma, which contains information both about the
# standard deviations sigma, a vector of length k, and the correlations between
# the parameters, which can be described as a k xk correlation matrix Omega. 
# It is usually easier to specify the correlation matrix Omega,
# and simply transform it to the covariance matrix Sigma, like so:

# Omega <- matrix(data = c(1, -0.9,
#                          -0.9, 1),
#                 ncol = 2)
# 
# mu <- c(10, 2)
# sigma <- c(1, 0.5)
# 
# # Transform from correlation matrix to covariance matrix
# Sigma <- diag(sigma) %*% Omega %*% diag(sigma)
# 
# correlated_parameters <- data.frame(
#                             subj_id = 1:N,
#                             alpha = NA,
#                             beta = NA
#                           )
# 
# correlated_parameters[, 2:3] <- MASS::mvrnorm(n = N,
#                                               mu = mu,
#                                               Sigma = Sigma)
# 
# plot(correlated_parameters[, 2:3])
# 
# 
# # We can then create a new set of data (with noise)
# multi_normal_data <- data.frame(
#     subj_id = gl(n = N, k = n_rep * length(x_series)),
#     rep = rep(x = 1:n_rep,
#               times = N * length(x_series)),
#     x = rep(x = sort(rep(x_series, n_rep)),
#             times = N),
#     y = NA
# )
# 
# multi_normal_data$y <-
#     linear(
#         alpha = correlated_parameters$alpha[multi_normal_data$subj_id],
#         beta = correlated_parameters$beta[multi_normal_data$subj_id],
#         x = normal_data$x
#         ) +
#     rnorm(n = length(multi_normal_data$y),
#         mean = 0,
#         sd = sd_y)
# 
# # Example plots - note that slopes get lower, the higher the intercept!
# ggplot(data = filter(multi_normal_data, subj_id %in% c(1, 2, 3, 4, 5, 6)),
#        mapping = aes(x = x,
#                      y = y,
#                      colour = subj_id)) +
#     geom_jitter(width = 0.1) +
#     geom_smooth(method = 'lm') +
#     theme_bw()
```

Et voilà! Here comes the question - can we model this in Stan?


```r
# samples to draw from each distribution
# N <- 20
# 
# true_parameters <- data.frame(
#         subj_id = 1:N,
#         alpha = rnorm(N, 100, 10),
#         beta = rnorm(N, 1, 0.2),
#         gamma = rnorm(N, 2, 1),
#         delta = rnorm(N, 10, 5)
#         )
# 
# true_parameters_long <- true_parameters %>%
#     pivot_longer(cols = !subj_id,
#                  names_to = 'parameter',
#                  values_to = 'value')
# 
# true_parameters_long$parameter <- factor(x = true_parameters_long$parameter,
#                               levels = c('alpha', 'beta', 'gamma', 'delta')
#                               )
# 
# four_PL <- function(alpha, beta, gamma, delta, x) {
#         delta + (alpha - delta) / (1 + exp(- beta * x + gamma))
# }
# 
# # There is a trick here somewhere to plot all graphs in one
# ggplot(data = true_parameters_long,
#        mapping = aes(x = value)
#        ) +
#     facet_wrap(~ parameter,
#                scales = 'free') +
#     geom_histogram() +
#     theme_bw()
# 
# # Set up a palette
# my_palette <- viridis::viridis(n = N)
# 
# # Set up canvas
# plot <- ggplot() +
#         xlim(-10, 10) +
#         theme_bw()
# 
# # Add the individual curves    
# for (n in 1:N) {
#     plot <- plot +
#     geom_function(fun = four_PL,
#                   args = list(alpha = true_parameters$alpha[n],
#                               beta = true_parameters$beta[n],
#                               gamma = true_parameters$gamma[n],
#                               delta = true_parameters$delta[n]
#                               ),
#                   colour = my_palette[n]
#                   )        
# }
# 
# plot
# 
# # We now know how to draw perfect data - but we want noisy data!
# # Let's add some noise and collect data in replicates at various x values
# 
# n_rep <- 3
# x_series <- seq(from = -10, to = 10, by = 1)
# 
# normal_data <- data.frame(
#     subj_id = gl(n = N, k = n_rep * length(x_series)),
#     rep = rep(x = 1:n_rep,
#               times = N * length(x_series)),
#     x = rep(x = sort(rep(x_series, n_rep)),
#             times = N),
#     y = NA
# )
# 
# # In this (simplistic) case we will have a common noise level on all data
# sd_y <- 5
# 
# normal_data$y <- four_PL(alpha = true_parameters$alpha[normal_data$subj_id],
#                          beta = true_parameters$beta[normal_data$subj_id],
#                          gamma = true_parameters$gamma[normal_data$subj_id],
#                          delta = true_parameters$delta[normal_data$subj_id],
#                          x = normal_data$x) +
#                  rnorm(n = length(normal_data$y),
#                        mean = 0,
#                        sd = sd_y)
# 
# ggplot(data = filter(normal_data, subj_id %in% c(1, 2, 3, 4, 5, 6)),
#        mapping = aes(x = x,
#                      y = y,
#                      colour = subj_id)) +
#     geom_jitter(width = 0.1)
```

Modelling data in a hierarchical structure.
Modelling data in multivariate normal hierarchical model





## Including Plots

You can also embed plots, for example:

![](introduction_to_multivariate_normal_files/figure-html/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.
