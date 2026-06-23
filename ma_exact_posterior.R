# =====================================================================
# ma_exact_posterior.R
#
# The EXACT (non-ABC) posterior for the MA(q) model, used as a ground
# truth against which to compare the ABC posteriors.
#
# An MA(q) series y_k = u_k + theta_1 u_{k-1} + ... + theta_q u_{k-q}
# with i.i.d. N(0,1) innovations is a zero-mean stationary Gaussian
# process,  y ~ N(0, Sigma(theta)),  where Sigma is the banded Toeplitz
# covariance with autocovariances
#       gamma_j = sum_k c_k c_{k+j},   c = (1, theta_1, ..., theta_q),
# (gamma_j = 0 for j > q).  This is the exact likelihood -- no ABC, no
# tolerance.  Combined with the invertibility prior from ma.R it gives
# the exact posterior, which we sample with an adaptive random-walk
# Metropolis-Hastings algorithm.
#
# Requires ma.R (log_density_prior_normalized, simulate_prior_maq).
# =====================================================================

# autocovariances gamma_0..gamma_q of MA(q) with coefficients theta
ma_autocov <- function(theta) {
  cc <- c(1, theta); q <- length(theta)
  vapply(0:q, function(j) sum(cc[seq_len(q + 1 - j)] * cc[seq_len(q + 1 - j) + j]),
         numeric(1))
}

# exact Gaussian log-likelihood of y under MA(q) with coefficients theta
ma_loglik_exact <- function(theta, y) {
  n <- length(y); q <- length(theta)
  g <- ma_autocov(theta)
  v <- c(g, numeric(max(0, n - q - 1)))[seq_len(n)]   # first row of Sigma
  Sigma <- stats::toeplitz(v)
  R <- tryCatch(chol(Sigma), error = function(e) NULL) # Sigma = R'R
  if (is.null(R)) return(-Inf)                         # not PD => reject
  z <- backsolve(R, y, transpose = TRUE)               # R'^{-1} y
  -0.5 * (n * log(2 * pi) + 2 * sum(log(diag(R))) + sum(z * z))
}

# exact log-posterior (up to a constant): invertibility prior + likelihood
ma_logpost_exact <- function(theta, y) {
  lp <- log_density_prior_normalized(theta)
  if (!is.finite(lp)) return(-Inf)
  lp + ma_loglik_exact(theta, y)
}

# ---------------------------------------------------------------------
# Adaptive random-walk Metropolis-Hastings on the exact MA(q) posterior.
# The proposal covariance is adapted from a trailing window of the chain
# and scaled by the Roberts-Gelman-Gilks factor 2.38^2/q (Haario-style
# adaptive Metropolis).  Returns post-burn-in samples (n_keep x q).
# ---------------------------------------------------------------------
ma_exact_mcmc <- function(y, q,
                          n_iter       = 40000,
                          burn_in      = 10000,
                          thin         = 1,
                          init         = NULL,
                          prop_sd0     = 0.05,
                          adapt_from   = 200,
                          adapt_window = 3000,
                          adapt_every  = 50) {
  d <- q
  lp_fun <- function(th) ma_logpost_exact(th, y)

  if (is.null(init)) {
    repeat { init <- as.numeric(simulate_prior_maq(1, q))
             if (is.finite(lp_fun(init))) break }
  }
  th <- init; lp <- lp_fun(th)
  chain <- matrix(NA_real_, n_iter, d)
  sd_scale <- 2.38^2 / d
  L <- diag(prop_sd0, d)                          # initial proposal Cholesky
  acc <- 0L

  for (i in seq_len(n_iter)) {
    prop <- th + as.vector(L %*% rnorm(d))
    lpp  <- lp_fun(prop)
    if (is.finite(lpp) && log(runif(1)) < lpp - lp) {
      th <- prop; lp <- lpp; acc <- acc + 1L
    }
    chain[i, ] <- th
    if (i >= adapt_from && i %% adapt_every == 0) {  # adapt proposal
      w0  <- max(1L, i - adapt_window)
      emp <- stats::cov(chain[w0:i, , drop = FALSE])
      L   <- tryCatch(t(chol(sd_scale * (emp + diag(1e-8, d)))),
                      error = function(e) L)
    }
  }

  keep <- seq(burn_in + 1L, n_iter, by = thin)
  list(samples  = chain[keep, , drop = FALSE],
       acc_rate = acc / n_iter,
       chain    = chain)
}

# ---------------------------------------------------------------------
# Reference: exact posterior on a 2-D grid (MA(2) only), for validating
# the MCMC.  Returns the grid, the (normalised) posterior, and its mean.
# ---------------------------------------------------------------------
ma_exact_grid_posterior <- function(y, lim1 = c(-2, 2), lim2 = c(-1, 1),
                                    ngrid = 161) {
  g1 <- seq(lim1[1], lim1[2], length.out = ngrid)
  g2 <- seq(lim2[1], lim2[2], length.out = ngrid)
  grid <- expand.grid(theta1 = g1, theta2 = g2)
  lp <- apply(grid, 1, function(th) ma_logpost_exact(as.numeric(th), y))
  w  <- exp(lp - max(lp[is.finite(lp)])); w[!is.finite(w)] <- 0; w <- w / sum(w)
  list(grid = grid, weight = w,
       mean = c(sum(w * grid$theta1), sum(w * grid$theta2)),
       g1 = g1, g2 = g2)
}
