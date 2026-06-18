# =====================================================================
# smc_utils.R
#
# Shared utilities for the SMC algorithms in the paper
# "Rare-event ABC-SMC^2".  These helpers are model-agnostic.
#
#   * weight normalisation / effective sample size
#   * multinomial (and systematic) resampling
#   * conditional effective sample size (CESS)
#   * adaptive tolerance selection by bisection on the CESS
#     (Section "Adapting the sequence of tolerances")
#   * adaptive number of MCMC moves
#     (Section "Adapting the number of MCMC moves", South et al. 2019)
# =====================================================================

# ------------------------------------------------------------------
# Numerically stable log-sum-exp.
# ------------------------------------------------------------------
log_sum_exp <- function(lx) {
  m <- max(lx)
  if (!is.finite(m)) return(m)         # all -Inf  ->  -Inf
  m + log(sum(exp(lx - m)))
}

# ------------------------------------------------------------------
# Normalise a vector of *log* unnormalised weights.
# Returns the normalised log-weights.
# ------------------------------------------------------------------
normalise_log_weights <- function(log_w) {
  log_w - log_sum_exp(log_w)
}

# ------------------------------------------------------------------
# Effective sample size from normalised log-weights:
#       ESS = 1 / sum_n w_n^2 .
# ------------------------------------------------------------------
ess_from_log_weights <- function(log_w_norm) {
  exp(-log_sum_exp(2 * log_w_norm))
}

# ------------------------------------------------------------------
# Multinomial resampling.  Returns a vector of ancestor indices,
# drawing index n with probability w_n (the M(.) of the paper).
# ------------------------------------------------------------------
resample_multinomial <- function(log_w_norm) {
  N <- length(log_w_norm)
  sample.int(N, size = N, replace = TRUE, prob = exp(log_w_norm))
}

# ------------------------------------------------------------------
# Systematic resampling (lower variance alternative, optional).
# ------------------------------------------------------------------
resample_systematic <- function(log_w_norm) {
  N <- length(log_w_norm)
  positions <- (runif(1) + 0:(N - 1)) / N
  cumw <- cumsum(exp(log_w_norm))
  cumw[N] <- 1                          # guard against rounding
  findInterval(positions, cumw, left.open = TRUE) + 1L
}

resample <- function(log_w_norm, scheme = c("multinomial", "systematic")) {
  scheme <- match.arg(scheme)
  if (scheme == "multinomial") resample_multinomial(log_w_norm)
  else                         resample_systematic(log_w_norm)
}

# ------------------------------------------------------------------
# Conditional effective sample size (Zhou et al. 2015).
#
#   CESS = N ( sum_m w_m r_m )^2 / ( sum_m w_m r_m^2 )
#
# where w_m are the (normalised) weights *before* reweighting and
# r_m the per-particle incremental weight (likelihood ratio).
# Inputs are on the log scale.
# ------------------------------------------------------------------
cess_from_logs <- function(log_w_prev_norm, log_ratio) {
  N <- length(log_ratio)
  num <- 2 * log_sum_exp(log_w_prev_norm + log_ratio)
  den <-     log_sum_exp(log_w_prev_norm + 2 * log_ratio)
  N * exp(num - den)
}

# ------------------------------------------------------------------
# Adaptive selection of the next tolerance epsilon_t by bisection.
#
# 'log_ratio_fn(epsilon)' must return the vector of per-particle log
# incremental weights  log r_m(epsilon)  that would result from
# moving from the current tolerance to 'epsilon'.  This is the cheap
# recomputation described in the paper (re-using already simulated
# x- or u-particles).
#
# We search within [eps_lower, eps_upper] for the epsilon whose CESS
# is closest to 'target' ( = beta * N ).  CESS increases with epsilon,
# so if even eps_lower already achieves CESS >= target we jump
# straight to eps_lower (typically the final desired tolerance).
# ------------------------------------------------------------------
adaptive_epsilon <- function(log_ratio_fn,
                             log_w_prev_norm,
                             target,
                             eps_lower,
                             eps_upper,
                             tol      = 1e-4,
                             max_iter = 100) {
  cess <- function(eps) cess_from_logs(log_w_prev_norm, log_ratio_fn(eps))

  # If we can already drop to the lower bound without over-depleting,
  # do so.
  if (cess(eps_lower) >= target) {
    return(eps_lower)
  }

  lo <- eps_lower
  hi <- eps_upper
  for (i in seq_len(max_iter)) {
    mid <- 0.5 * (lo + hi)
    cm  <- cess(mid)
    if (cm < target) lo <- mid else hi <- mid   # CESS increasing in eps
    if (hi - lo < tol) break
  }
  0.5 * (lo + hi)
}

# ------------------------------------------------------------------
# Weighted mean and weighted *sample* covariance of a population of
# particles.
#
# 'particles' is a list, each element a numeric vector (length d; d = 1
# for scalar theta).  'weights' are non-negative and need not be
# normalised.  The covariance uses the unbiased weighted-sample
# estimator (as in stats::cov.wt, method = "unbiased"):
#
#   Sigma = ( sum_m w_m (x_m - mu)(x_m - mu)^T ) / ( 1 - sum_m w_m^2 ),
#
# with the weights w_m normalised to sum to one.  Returns the weighted
# mean, the d x d weighted sample covariance, and the per-component
# weighted standard deviations.
# ------------------------------------------------------------------
weighted_moments <- function(particles, weights) {
  X  <- do.call(rbind, lapply(particles, as.numeric))   # N x d
  w  <- weights / sum(weights)
  mu <- colSums(w * X)
  Xc <- sweep(X, 2L, mu)
  denom <- 1 - sum(w^2)                                  # unbiased correction
  if (denom < 1e-8) denom <- 1                           # guard weight collapse
  Sigma <- crossprod(Xc, w * Xc) / denom                 # weighted sample cov.
  list(mean = mu, cov = Sigma, sd = sqrt(pmax(diag(Sigma), 0)))
}

# ------------------------------------------------------------------
# Build a symmetric multivariate-Gaussian random-walk proposal with a
# given covariance matrix:  theta* = theta + L z,  z ~ N(0, I_d),
# L L^T = cov.  This is a genuine multivariate proposal: off-diagonal
# (cross-component) correlations of 'cov' are respected via L.
#
# Because the covariance is held fixed across the move step, the
# proposal is symmetric and its density cancels in the MH acceptance
# ratio, so only the sampler is needed.  A small jitter keeps the
# Cholesky factor well-defined when the covariance is (near-)singular.
# ------------------------------------------------------------------
make_gaussian_rw_proposal <- function(cov, scale = 1) {
  cov <- scale^2 * as.matrix(cov)
  d   <- nrow(cov)
  jitter <- 1e-12 * max(diag(cov), 1)
  L <- tryCatch(t(chol(cov + diag(jitter, d))),
                error = function(e) diag(sqrt(pmax(diag(cov), 0)), d))
  function(theta) as.numeric(theta) + as.numeric(L %*% rnorm(d))
}

# ------------------------------------------------------------------
# Adaptive number of MCMC sweeps (South et al. 2019):
#
#   ceil( log(c) / log(1 - p_acc) )
#
# chosen so that there is an estimated probability (1 - c) that each
# particle is moved at least once.  Guards against p_acc in {0, 1}.
# ------------------------------------------------------------------
adaptive_num_mcmc <- function(p_acc, c = 0.2, max_moves = 100L) {
  if (p_acc <= 0)  return(max_moves)            # nothing moves: do the max
  if (p_acc >= 1)  return(1L)                   # everything moves: one is enough
  n <- ceiling(log(c) / log(1 - p_acc))
  as.integer(min(max(n, 1L), max_moves))
}
