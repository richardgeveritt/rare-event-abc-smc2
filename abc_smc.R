# =====================================================================
# abc_smc.R
#
# Algorithm "ABC-SMC" (alg:abc-smc) from the paper, in the
# Del Moral et al. (2012) form: a sequence of ABC posteriors with
# decreasing tolerances, the likelihood at each theta estimated from
# Nx simulations,
#       lhat_s(y | theta) = (1/Nx) sum_n P_eps_s(y | x^n) .
#
# This is the comparator for RE-ABC-SMC^2.  It does NOT use the
# reparameterisation: it simulates x directly.
#
# ---------------------------------------------------------------------
# Required model interface:
#
#   model$rprior(N)            -> list of N draws theta ~ p
#   model$dprior(theta)        -> log p(theta)                  (scalar)
#   model$rproposal(theta)     -> proposed theta* ~ q(. | theta)
#   model$dproposal(to, from)  -> log q(to | from)              (scalar)
#   model$simulate(theta, Nx)  -> list of Nx draws x ~ f(. | theta)
#   model$log_abc_kernel(x, epsilon) -> log P_eps(y | x)        (scalar)
#
# 'y' is assumed captured inside log_abc_kernel.
# =====================================================================

source("R/smc_utils.R")

# log of the Nx-point ABC likelihood estimate at tolerance epsilon,
# given an already-simulated list of x's:
#       log( (1/Nx) sum_n P_eps(y | x^n) ).
.abc_log_lhat <- function(x_list, epsilon, model) {
  lk <- vapply(x_list, function(xx) model$log_abc_kernel(xx, epsilon),
               numeric(1))
  log_sum_exp(lk) - log(length(x_list))
}

# ------------------------------------------------------------------
# One ABC-MCMC step on a single theta-particle (lines 4-7 of
# algorithm abc-mcmc, used as the SMC move kernel K_t).  The current
# likelihood estimate is cached and passed/returned to avoid recompute.
#
# If 'rprop' is supplied it is used as a *symmetric* proposal (e.g. the
# adaptive Gaussian random walk whose covariance is the weighted
# covariance of the population): its density then cancels in the MH
# ratio.  Otherwise the model's own rproposal/dproposal are used.
# ------------------------------------------------------------------
.abc_mcmc_step <- function(theta, x_list, log_lhat_cur, epsilon, Nx, model,
                           rprop = NULL) {
  if (is.null(rprop)) {
    theta_prop <- model$rproposal(theta)
    log_q_ratio <- model$dproposal(theta, theta_prop) -
                   model$dproposal(theta_prop, theta)
  } else {
    theta_prop  <- rprop(theta)
    log_q_ratio <- 0                       # symmetric proposal
  }
  x_prop     <- model$simulate(theta_prop, Nx)
  log_lhat_prop <- .abc_log_lhat(x_prop, epsilon, model)

  log_alpha <- (model$dprior(theta_prop) - model$dprior(theta)) +
               log_q_ratio +
               (log_lhat_prop - log_lhat_cur)

  if (log(runif(1)) < log_alpha) {
    list(theta = theta_prop, x = x_prop, log_lhat = log_lhat_prop, accepted = TRUE)
  } else {
    list(theta = theta, x = x_list, log_lhat = log_lhat_cur, accepted = FALSE)
  }
}

# ------------------------------------------------------------------
# ABC-SMC.
#
# Tolerances can be supplied explicitly via 'epsilon_schedule', or
# chosen adaptively (the default) by targeting CESS = beta * N_theta
# down to a final tolerance 'epsilon_final'.
#
# Returns a list with the final weighted theta-population, the log
# model-evidence estimate, and per-iteration diagnostics.
# ------------------------------------------------------------------
abc_smc <- function(model,
                    N_theta,
                    Nx,
                    epsilon_schedule = NULL,   # explicit eps_1 > ... > eps_T
                    epsilon_final    = NULL,   # adaptive: stop here
                    epsilon_start    = Inf,    # adaptive: upper bound for eps_1
                    beta             = 0.9,    # adaptive: target CESS fraction
                    alpha            = 0.5,    # resample if ESS < alpha * N_theta
                    max_steps        = 200L,   # safety cap for adaptive mode
                    resample_scheme  = "multinomial",
                    adapt_nmoves     = TRUE,
                    c_move           = 0.2,
                    max_moves        = 100L,
                    adapt_theta_proposal = TRUE,  # adaptive Gaussian RW on theta
                    proposal_scale       = NULL,  # NULL => 2.38/sqrt(d) (optimal); else a fixed scale
                    record_history   = FALSE,     # keep the weighted theta-population each iteration
                    verbose          = FALSE) {

  adaptive <- is.null(epsilon_schedule)
  if (!adaptive) {
    epsilon_final <- epsilon_schedule[length(epsilon_schedule)]
  } else if (is.null(epsilon_final)) {
    stop("Provide either 'epsilon_schedule' or 'epsilon_final'.")
  }

  ## --- initialise (lines 1-2): theta_0 ~ p, simulate x ------------
  theta <- model$rprior(N_theta)
  if (!is.list(theta)) theta <- split(theta, seq_len(N_theta))
  x      <- lapply(theta, function(th) model$simulate(th, Nx))
  log_w  <- rep(-log(N_theta), N_theta)

  eps_prev <- NA_real_
  log_lhat_prev <- rep(0, N_theta)        # lhat at eps_prev per particle

  log_evidence  <- 0
  eps_history   <- numeric(0)
  ess_history   <- numeric(0)
  accrate_hist  <- numeric(0)
  history       <- list()

  T_total <- if (adaptive) max_steps else length(epsilon_schedule)

  for (t in seq_len(T_total)) {

    ## --- choose epsilon_t -----------------------------------------
    if (adaptive) {
      # log incremental weight per particle as a function of candidate eps
      log_ratio_fn <- function(eps) {
        cur <- vapply(x, function(xl) .abc_log_lhat(xl, eps, model), numeric(1))
        if (t == 1L) cur else cur - log_lhat_prev
      }
      eps_t <- if (t == 1L) {
        # first step: target CESS from the uniform-weight prior population
        adaptive_epsilon(log_ratio_fn, log_w, beta * N_theta,
                         eps_lower = epsilon_final,
                         eps_upper = if (is.finite(epsilon_start)) epsilon_start
                                     else .abc_eps_upper(x, model))
      } else {
        adaptive_epsilon(log_ratio_fn, log_w, beta * N_theta,
                         eps_lower = epsilon_final, eps_upper = eps_prev)
      }
    } else {
      eps_t <- epsilon_schedule[t]
    }

    ## --- reweight (lines 3-7) -------------------------------------
    log_lhat_t <- vapply(x, function(xl) .abc_log_lhat(xl, eps_t, model),
                         numeric(1))
    log_incr   <- if (t == 1L) log_lhat_t else log_lhat_t - log_lhat_prev
    log_wtilde <- log_w + log_incr

    ## model-evidence contribution: sum_m w_{t-1}^m * incr_m
    log_evidence <- log_evidence + log_sum_exp(log_wtilde)

    ## --- normalise (line 8) ---------------------------------------
    log_w <- normalise_log_weights(log_wtilde)
    ess   <- ess_from_log_weights(log_w)

    eps_history <- c(eps_history, eps_t)
    ess_history <- c(ess_history, ess)

    ## record the weighted theta-population representing target t
    if (record_history) {
      history[[t]] <- list(
        theta   = do.call(rbind, lapply(theta, as.numeric)),
        log_w   = log_w,
        epsilon = eps_t)
    }

    ## --- resample & move if degenerate (lines 9-12) ---------------
    acc_rate <- NA_real_
    if (ess < alpha * N_theta) {
      ## adaptive proposal: weighted covariance of theta *before*
      ## resampling (i.e. using the current normalised weights), scaled by
      ## the Roberts-Gelman-Gilks optimal factor 2.38/sqrt(d) by default.
      rprop <- if (adapt_theta_proposal) {
        wm <- weighted_moments(theta, exp(log_w))
        scl <- if (is.null(proposal_scale)) 2.38 / sqrt(nrow(wm$cov)) else proposal_scale
        make_gaussian_rw_proposal(wm$cov, scale = scl)
      } else NULL

      anc      <- resample(log_w, resample_scheme)
      theta    <- theta[anc]
      x        <- x[anc]
      log_lhat_t <- log_lhat_t[anc]
      log_w    <- rep(-log(N_theta), N_theta)

      ## ABC-MCMC moves with adaptive number of sweeps
      mh_sweep <- function() {
        acc <- 0L
        for (m in seq_len(N_theta)) {
          res <- .abc_mcmc_step(theta[[m]], x[[m]], log_lhat_t[m],
                                eps_t, Nx, model, rprop = rprop)
          theta[[m]]    <<- res$theta
          x[[m]]        <<- res$x
          log_lhat_t[m] <<- res$log_lhat
          acc <- acc + res$accepted
        }
        acc / N_theta
      }
      acc_rate <- mh_sweep()
      n_sweeps <- if (adapt_nmoves) adaptive_num_mcmc(acc_rate, c_move, max_moves) else 1L
      if (n_sweeps > 1L) for (s in seq_len(n_sweeps - 1L)) mh_sweep()
    }
    accrate_hist <- c(accrate_hist, acc_rate)

    if (verbose) {
      cat(sprintf("[ABC-SMC] t=%d  eps=%.4g  ESS=%.1f  acc=%s\n",
                  t, eps_t, ess,
                  if (is.na(acc_rate)) "-" else sprintf("%.2f", acc_rate)))
    }

    eps_prev      <- eps_t
    log_lhat_prev <- log_lhat_t

    if (adaptive && eps_t <= epsilon_final + 1e-12) break
  }

  list(
    theta        = theta,
    log_weights  = log_w,
    weights      = exp(log_w),
    log_evidence = log_evidence,
    epsilon      = eps_history,
    ess          = ess_history,
    acc_rate     = accrate_hist,
    n_iterations = length(eps_history),
    history      = if (record_history) history else NULL
  )
}

# A crude upper bound for the first adaptive tolerance when
# epsilon_start = Inf: the largest single-point distance currently seen
# (so that CESS at this eps is ~ N_theta).  Uses the kernel monotonicity
# only loosely; users may prefer to pass an explicit 'epsilon_start'.
.abc_eps_upper <- function(x, model) {
  # Probe increasing tolerances until CESS is essentially N (kernel ~ const).
  # Falls back to a large finite value.
  eps <- 1
  for (k in 1:60) {
    lk <- unlist(lapply(x, function(xl)
      vapply(xl, function(xx) model$log_abc_kernel(xx, eps), numeric(1))))
    if (all(is.finite(lk)) && diff(range(lk)) < 1e-6) return(eps)
    eps <- eps * 2
  }
  eps
}
