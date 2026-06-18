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
