# =====================================================================
# re_abc_smc2.R
#
# Algorithm "Rare event ABC-SMC^2 (RE-ABC-SMC^2)" (alg:abc-smc2).
#
# An external SMC sampler over theta, in which the ABC likelihood for
# each theta is estimated by an *internal* rare event SMC over u
# (alg:rare-event-smc, in rare_event_smc.R).  Each external
# theta-particle carries the state of its own inner SMC; at external
# iteration t every inner SMC is advanced by one tolerance eps_t, and
# the external incremental weight is
#       sum_n wtilde^{n,m}_t  =  lhat_t(y|theta) / lhat_{t-1}(y|theta).
#
# The external move is the particle-MCMC move of Prangle (2016): for
# each particle draw an ancestor i*, propose theta* ~ q_t(.|theta_{i*}),
# run a fresh rare event SMC up to eps_t conditional on theta*, and
# accept with the marginal-likelihood ratio (a pseudo-marginal /
# SMC^2-style acceptance).
#
# This uses the reparameterised model interface documented in
# rare_event_smc.R, plus the theta-level prior / proposal:
#
#   model$rprior(N), model$dprior(theta),
#   model$rproposal(theta), model$dproposal(to, from),
#   model$rphi, model$dphi, model$H, model$log_abc_kernel,
#   model$ru_move, model$du_move
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")

# log incremental external weight log r_m(eps) for particle m's inner
# SMC, *without* advancing it (used for adaptive tolerance choice):
#       log sum_n w^{n,m} * P_eps / P_{eps_prev}.
.re2_log_incr_candidate <- function(state, eps, model) {
  log_sum_exp(state$log_w + re_smc_log_incremental(state, eps, model))
}

# Probe an upper bound for the first tolerance (when eps_start = Inf):
# the smallest eps for which every inner kernel value is ~constant.
.re2_eps_upper <- function(states, model) {
  eps <- 1
  for (k in 1:60) {
    lk <- unlist(lapply(states, function(st)
      vapply(st$x, function(xx) model$log_abc_kernel(xx, eps), numeric(1))))
    if (all(is.finite(lk)) && diff(range(lk)) < 1e-6) return(eps)
    eps <- eps * 2
  }
  eps
}

re_abc_smc2 <- function(model,
                        N_theta,
                        Nu,
                        epsilon_schedule = NULL, # explicit eps_1 > ... > eps_T
                        epsilon_final    = NULL, # adaptive: stop here
                        epsilon_start    = Inf,  # adaptive: upper bound for eps_1
                        beta             = 0.9,  # adaptive: target CESS fraction
                        alpha            = 0.5,  # external resample if ESS < alpha*N_theta
                        inner_alpha      = 0.5,  # inner resample threshold
                        max_steps        = 200L, # safety cap (adaptive mode)
                        resample_scheme  = "multinomial",
                        move             = "pcn-mala",  # inner u-move: "pcn-mala" | "mh" | "gaussian"
                        move_step0       = 1,     # initial pCN-MALA step size
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
  move <- .resolve_move(move, model)   # fall back to "mh" if model lacks the pieces

  inner_step <- function(state, eps) {
    re_smc_step(state, eps, model,
                alpha = inner_alpha, resample_scheme = resample_scheme,
                move = move, adapt_nmoves = adapt_nmoves, c_move = c_move,
                max_moves = max_moves)
  }
  inner_run <- function(theta, schedule) {
    re_smc_run(model, theta, Nu, schedule,
               alpha = inner_alpha, resample_scheme = resample_scheme,
               move = move, move_step0 = move_step0, adapt_nmoves = adapt_nmoves,
               c_move = c_move, max_moves = max_moves)
  }

  ## --- initialise (lines 1-3): theta_0 ~ p, u_0 ~ phi -------------
  theta <- model$rprior(N_theta)
  if (!is.list(theta)) theta <- split(theta, seq_len(N_theta))
  states <- lapply(theta,                                           # inner SMCs
                   function(th) re_smc_init(model, th, Nu, move_step0 = move_step0))
  log_w  <- rep(-log(N_theta), N_theta)

  eps_prev      <- NA_real_
  realised_sched <- numeric(0)        # the tolerance schedule used so far
  log_evidence  <- 0
  eps_history   <- numeric(0)
  ess_history   <- numeric(0)
  accrate_hist  <- numeric(0)
  history       <- list()

  T_total <- if (adaptive) max_steps else length(epsilon_schedule)

  for (t in seq_len(T_total)) {

    ## --- choose eps_t (Section "Adapting the sequence of tolerances")
    if (adaptive) {
      log_ratio_fn <- function(eps)
        vapply(states, function(st) .re2_log_incr_candidate(st, eps, model),
               numeric(1))
      eps_t <- if (t == 1L) {
        adaptive_epsilon(log_ratio_fn, log_w, beta * N_theta,
                         eps_lower = epsilon_final,
                         eps_upper = if (is.finite(epsilon_start)) epsilon_start
                                     else .re2_eps_upper(states, model))
      } else {
        adaptive_epsilon(log_ratio_fn, log_w, beta * N_theta,
                         eps_lower = epsilon_final, eps_upper = eps_prev)
      }
    } else {
      eps_t <- epsilon_schedule[t]
    }
    realised_sched <- c(realised_sched, eps_t)

    ## --- reweight: advance every inner SMC one step to eps_t -------
    log_incr <- numeric(N_theta)
    for (m in seq_len(N_theta)) {
      states[[m]]  <- inner_step(states[[m]], eps_t)
      log_incr[m]  <- states[[m]]$last_log_incr   # log sum_n wtilde^{n,m}_t
    }
    log_wtilde   <- log_w + log_incr
    log_evidence <- log_evidence + log_sum_exp(log_wtilde)

    ## --- normalise ------------------------------------------------
    log_w <- normalise_log_weights(log_wtilde)
    ess   <- ess_from_log_weights(log_w)

    eps_history <- c(eps_history, eps_t)
    ess_history <- c(ess_history, ess)

    ## record the weighted theta-population representing target t
    if (record_history) {
      history[[t]] <- list(
        theta   = do.call(rbind, lapply(states, function(st) as.numeric(st$theta))),
        log_w   = log_w,
        epsilon = eps_t)
    }

    ## --- resample & move if degenerate ----------------------------
    acc_rate <- NA_real_
    if (ess < alpha * N_theta) {
      ## adaptive proposal: weighted covariance of theta *before*
      ## resampling (i.e. using the current normalised weights), scaled by
      ## the Roberts-Gelman-Gilks optimal factor 2.38/sqrt(d) by default.
      rprop <- if (adapt_theta_proposal) {
        wm <- weighted_moments(lapply(states, function(st) st$theta), exp(log_w))
        scl <- if (is.null(proposal_scale)) 2.38 / sqrt(nrow(wm$cov)) else proposal_scale
        make_gaussian_rw_proposal(wm$cov, scale = scl)
      } else NULL

      new_theta  <- vector("list", N_theta)
      new_states <- vector("list", N_theta)
      accepted   <- logical(N_theta)

      for (m in seq_len(N_theta)) {
        istar  <- resample_multinomial_one(log_w)
        th_old <- states[[istar]]$theta
        st_old <- states[[istar]]
        loglik_old <- st_old$log_lik           # log prod_t sum_n wtilde for i*

        if (is.null(rprop)) {
          th_star <- model$rproposal(th_old)
          log_q_ratio <- model$dproposal(th_old, th_star) -
                         model$dproposal(th_star, th_old)
        } else {
          th_star <- rprop(th_old)
          log_q_ratio <- 0                     # symmetric proposal
        }
        st_star <- inner_run(th_star, realised_sched)  # fresh inner SMC up to eps_t
        loglik_star <- st_star$log_lik

        log_alpha <- (model$dprior(th_star) - model$dprior(th_old)) +
                     log_q_ratio +
                     (loglik_star - loglik_old)

        if (log(runif(1)) < log_alpha) {
          new_theta[[m]]  <- th_star
          new_states[[m]] <- st_star
          accepted[m]     <- TRUE
        } else {
          new_theta[[m]]  <- th_old
          new_states[[m]] <- st_old
          accepted[m]     <- FALSE
        }
      }
      theta    <- new_theta
      states   <- new_states
      log_w    <- rep(-log(N_theta), N_theta)
      acc_rate <- mean(accepted)
    } else {
      ## keep theta in sync with inner states (unchanged this iteration)
      theta <- lapply(states, function(st) st$theta)
    }
    accrate_hist <- c(accrate_hist, acc_rate)

    if (verbose) {
      cat(sprintf("[RE-ABC-SMC2] t=%d  eps=%.4g  ESS=%.1f  acc=%s\n",
                  t, eps_t, ess,
                  if (is.na(acc_rate)) "-" else sprintf("%.2f", acc_rate)))
    }

    eps_prev <- eps_t
    if (adaptive && eps_t <= epsilon_final + 1e-12) break
  }

  theta <- lapply(states, function(st) st$theta)
  list(
    theta            = theta,
    log_weights      = log_w,
    weights          = exp(log_w),
    log_evidence     = log_evidence,
    epsilon          = eps_history,
    epsilon_schedule = realised_sched,
    ess              = ess_history,
    acc_rate         = accrate_hist,
    states           = states,        # inner SMC states (u-particles etc.)
    n_iterations     = length(eps_history),
    history          = if (record_history) history else NULL
  )
}

# Draw a single ancestor index i* ~ M({omega_i}).
resample_multinomial_one <- function(log_w_norm) {
  sample.int(length(log_w_norm), size = 1L, prob = exp(log_w_norm))
}
