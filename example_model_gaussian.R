# =====================================================================
# example_model_gaussian.R
#
# A throw-away example model used only to exercise / sanity-check the
# algorithm code.  It implements the truncated-Gaussian example of the
# paper's results section: y is d i.i.d. draws from a half-normal
# (truncated N(0, sigma), lower bound 0); the goal is to infer
# theta = sigma, with prior U(0, 10).
#
# Reparameterisation:  u = (z_1, ..., z_d), z_i ~ N(0,1) i.i.d.
#                      H(u, sigma) = sigma * |u|   (half-normal samples)
#                      phi(u | theta) = prod N(z_i; 0, 1)
#
# ABC kernel: Gaussian on the Euclidean distance between order
# statistics of x and y.
#
# *** This file is only for testing; replace it with the real model. ***
# =====================================================================

make_gaussian_model <- function(y, prior_lower = 0, prior_upper = 10,
                                 rw_sd_theta = 0.4, rw_sd_u = 0.5) {
  d        <- length(y)
  y_sorted <- sort(y)

  distance <- function(x) sqrt(sum((sort(x) - y_sorted)^2))

  list(
    y = y,

    # ---- theta level -------------------------------------------------
    rprior = function(N) as.list(runif(N, prior_lower, prior_upper)),
    dprior = function(theta) {
      if (theta > prior_lower && theta < prior_upper)
        -log(prior_upper - prior_lower) else -Inf
    },
    rproposal = function(theta) theta + rnorm(1, 0, rw_sd_theta),
    dproposal = function(to, from) dnorm(to, from, rw_sd_theta, log = TRUE),

    # ---- direct simulation (for ABC-SMC) -----------------------------
    simulate = function(theta, Nx) {
      lapply(seq_len(Nx), function(i) theta * abs(rnorm(d)))
    },

    # ---- reparameterised pieces (for RE-ABC-SMC^2 / rare event SMC) ---
    rphi = function(theta, Nu) lapply(seq_len(Nu), function(i) rnorm(d)),
    dphi = function(u, theta) sum(dnorm(u, log = TRUE)),
    H    = function(u, theta) theta * abs(u),

    # random-walk MCMC move on u (symmetric proposal -> du_move = 0)
    ru_move = function(u, theta) u + rnorm(length(u), 0, rw_sd_u),
    du_move = function(to, from, theta) 0,

    # ---- ABC kernel: Gaussian on the order-statistic distance --------
    log_abc_kernel = function(x, epsilon) {
      -0.5 * (distance(x) / epsilon)^2
    }
  )
}
