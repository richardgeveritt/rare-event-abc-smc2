# =====================================================================
# rare_event_smc.R
#
# Algorithm "Rare event SMC" (alg:rare-event-smc) from the paper.
#
# This estimates the ABC likelihood
#       l(y | theta) = \int P_eps(y | H(u, theta)) phi(u | theta) du
# for a *single* theta, by running an SMC sampler over u through a
# decreasing sequence of tolerances  eps_1 > ... > eps_T = eps.
#
# The code is written so that the same engine serves two purposes:
#   (1) a stand-alone run over a whole tolerance schedule
#       ( re_smc_run() ),  used e.g. for the proposed theta* inside
#       the ABC-SMC^2 external move; and
#   (2) a persistent state that is advanced one tolerance at a time
#       ( re_smc_init() / re_smc_step() ), used for the inner SMCs
#       that are carried by each external theta-particle in ABC-SMC^2.
#
# ---------------------------------------------------------------------
# Required model interface (reparameterised / "unpacked" simulator):
#
#   model$rphi(theta, Nu)      -> list of Nu draws u ~ phi(. | theta)
#   model$dphi(u, theta)       -> log phi(u | theta)            (scalar)
#   model$H(u, theta)          -> x  (deterministic transform)  (single)
#   model$log_abc_kernel(x, epsilon) -> log P_eps(y | x)        (scalar)
#   model$ru_move(u, theta)    -> proposed u* for the MCMC move on u
#   model$du_move(u_to, u_from, theta) -> log proposal density
#                                 (return 0 for a symmetric proposal)
#
# 'y' is assumed to be captured inside log_abc_kernel / H.
# =====================================================================

source("R/smc_utils.R")

# ------------------------------------------------------------------
# One Metropolis-Hastings step for a single u-particle, targeting
#   P_eps(y | H(u, theta)) phi(u | theta).
# ------------------------------------------------------------------
.re_mh_step_u <- function(u, x, log_kernel_cur, theta, epsilon, model) {
  u_prop <- model$ru_move(u, theta)
  x_prop <- model$H(u_prop, theta)
  log_kernel_prop <- model$log_abc_kernel(x_prop, epsilon)

  log_alpha <- (log_kernel_prop + model$dphi(u_prop, theta)) -
               (log_kernel_cur  + model$dphi(u,      theta)) +
               model$du_move(u, u_prop, theta) -
               model$du_move(u_prop, u, theta)

  if (log(runif(1)) < log_alpha) {
    list(u = u_prop, x = x_prop, log_kernel = log_kernel_prop, accepted = TRUE)
  } else {
    list(u = u, x = x, log_kernel = log_kernel_cur, accepted = FALSE)
  }
}

# ------------------------------------------------------------------
# One pCN-MALA (infinity-MALA / Crank-Nicolson Langevin) step for a
# single u-particle, targeting P_eps(y|H(u,theta)) phi(u).  Requires the
# model to supply grad_loglik_u = d/du log P_eps(y|H(u,theta)).  The
# a^2 + b^2 = 1 construction makes the no-gradient limit exactly pCN; it
# is most efficient when phi = N(0, I) (true for ma.R / lv.R), but the
# full Metropolis-Hastings correction below is valid for any phi.
# ------------------------------------------------------------------
.re_pcn_mala_step_u <- function(u, x, log_kernel_cur, gll_cur,
                                theta, epsilon, delta, model) {
  a  <- (1 - delta / 4) / (1 + delta / 4)
  b  <- sqrt(delta) / (1 + delta / 4)
  cc <- (delta / 2) / (1 + delta / 4)

  mu      <- a * u + cc * gll_cur
  u_prop  <- mu + b * rnorm(length(u))
  x_prop  <- model$H(u_prop, theta)
  log_kernel_prop <- model$log_abc_kernel(x_prop, epsilon)
  gll_prop <- model$grad_loglik_u(u_prop, theta, epsilon)
  mu_rev  <- a * u_prop + cc * gll_prop

  log_alpha <- (log_kernel_prop + model$dphi(u_prop, theta)) -
               (log_kernel_cur  + model$dphi(u,      theta)) +
               (-sum((u      - mu_rev)^2) / (2 * b^2)) -      # log q(u | u_prop)
               (-sum((u_prop - mu    )^2) / (2 * b^2))        # log q(u_prop | u)

  if (log(runif(1)) < log_alpha) {
    list(u = u_prop, x = x_prop, log_kernel = log_kernel_prop,
         gll = gll_prop, accepted = TRUE)
  } else {
    list(u = u, x = x, log_kernel = log_kernel_cur, gll = gll_cur, accepted = FALSE)
  }
}

# ------------------------------------------------------------------
# Resolve the requested move against the model's capabilities, falling
# back (silently) to the generic MH move when the gradient / exact
# sampler it needs is absent.  This lets "pcn-mala" be the default while
# gradient-free models (e.g. lv.R, example_model_gaussian.R) still run.
# ------------------------------------------------------------------
.resolve_move <- function(move, model) {
  if (move == "pcn-mala" && is.null(model$grad_loglik_u))    return("mh")
  if (move == "gaussian" && is.null(model$rtarget_gaussian)) return("mh")
  move
}

# ------------------------------------------------------------------
# Move all u-particles.  'move' selects the kernel K_t:
#   "mh"        generic Metropolis-Hastings via model$ru_move/du_move
#               (e.g. the pCN move defined in ma.R / lv.R)
#   "pcn-mala"  gradient-based pCN-MALA (model$grad_loglik_u required);
#               the step size delta (state$move_step) is adapted towards
#               an acceptance of 0.574 and persists across SMC steps
#   "gaussian"  replace every particle with an independent EXACT draw
#               from the Gaussian target (model$rtarget_gaussian
#               required) -- a perfect move, no MCMC
# For the MCMC moves the number of sweeps is chosen adaptively (Section
# "Adapting the number of MCMC moves").
# ------------------------------------------------------------------
.re_move_all <- function(state, epsilon, model, move,
                         adapt_nmoves, c_move, max_moves) {
  Nu <- length(state$u)

  # ---- exact independent draw from the Gaussian target -------------
  if (move == "gaussian") {
    if (is.null(model$rtarget_gaussian))
      stop("move = 'gaussian' requires the model to provide rtarget_gaussian().")
    u <- model$rtarget_gaussian(state$theta, epsilon, Nu)
    if (!is.list(u)) u <- split(u, seq_len(Nu))
    state$u <- u
    state$x <- lapply(u, function(uu) model$H(uu, state$theta))
    state$log_kernel <- vapply(state$x,
      function(xx) model$log_abc_kernel(xx, epsilon), numeric(1))
    state$last_acc_rate <- 1
    return(state)
  }

  # ---- pCN-MALA ----------------------------------------------------
  if (move == "pcn-mala") {
    if (is.null(model$grad_loglik_u))
      stop("move = 'pcn-mala' requires the model to provide grad_loglik_u().")
    gll <- lapply(state$u,
      function(uu) model$grad_loglik_u(uu, state$theta, epsilon))
    one_sweep <- function(delta) {
      accept <- 0L
      for (n in seq_len(Nu)) {
        res <- .re_pcn_mala_step_u(state$u[[n]], state$x[[n]], state$log_kernel[n],
                                   gll[[n]], state$theta, epsilon, delta, model)
        state$u[[n]]        <<- res$u
        state$x[[n]]        <<- res$x
        state$log_kernel[n] <<- res$log_kernel
        gll[[n]]            <<- res$gll
        accept <- accept + res$accepted
      }
      accept / Nu
    }
    delta <- state$move_step
    p_acc <- one_sweep(delta)
    delta <- min(max(delta * exp(0.5 * (p_acc - 0.574)), 1e-6), 3.99)  # adapt
    state$move_step <- delta
    n_sweeps <- if (adapt_nmoves) adaptive_num_mcmc(p_acc, c_move, max_moves) else 1L
    if (n_sweeps > 1L) for (s in seq_len(n_sweeps - 1L)) one_sweep(delta)
    state$last_acc_rate <- p_acc
    return(state)
  }

  # ---- default: generic Metropolis-Hastings move -------------------
  one_sweep <- function() {
    accept <- 0L
    for (n in seq_len(Nu)) {
      res <- .re_mh_step_u(state$u[[n]], state$x[[n]], state$log_kernel[n],
                           state$theta, epsilon, model)
      state$u[[n]]          <<- res$u
      state$x[[n]]          <<- res$x
      state$log_kernel[n]   <<- res$log_kernel
      accept <- accept + res$accepted
    }
    accept / Nu
  }

  p_acc <- one_sweep()                                   # always at least one
  n_sweeps <- if (adapt_nmoves) adaptive_num_mcmc(p_acc, c_move, max_moves) else 1L
  if (n_sweeps > 1L) for (s in seq_len(n_sweeps - 1L)) one_sweep()

  state$last_acc_rate <- p_acc
  state
}

# ------------------------------------------------------------------
# Initialise the inner SMC (line 1 of algorithm rare-event-smc):
# simulate u_0 ~ phi(. | theta), uniform weights, no step taken yet.
# ------------------------------------------------------------------
re_smc_init <- function(model, theta, Nu, move_step0 = 1) {
  u <- model$rphi(theta, Nu)
  if (!is.list(u)) u <- split(u, seq_len(Nu))   # accept matrix/vector returns
  x <- lapply(u, function(uu) model$H(uu, theta))
  list(
    theta        = theta,
    Nu           = Nu,
    u            = u,
    x            = x,
    log_kernel   = rep(NA_real_, Nu),  # log P_{eps_last}(y | x_n), filled on 1st step
    log_w        = rep(-log(Nu), Nu),  # normalised log weights
    last_epsilon = NA_real_,
    t            = 0L,                  # number of tolerance steps taken
    log_lik      = 0,                   # running log of prod_t sum_n wtilde
    move_step    = move_step0,          # pCN-MALA step size (adapted, persists)
    last_acc_rate = NA_real_
  )
}

# ------------------------------------------------------------------
# Per-particle log incremental weight log r_n(epsilon) that would be
# obtained by reweighting the *current* u-particles to 'epsilon'.
# (lines 3-9 of algorithm rare-event-smc; the cheap recomputation used
#  for adaptive tolerance selection.)
#
#   t == 0  (first step):  log r_n =                 log P_eps(y | x_n)
#   t  > 0               :  log r_n = log P_eps(...) - log P_{eps_prev}(...)
# ------------------------------------------------------------------
re_smc_log_incremental <- function(state, epsilon, model) {
  lk_new <- vapply(state$x, function(xx) model$log_abc_kernel(xx, epsilon),
                   numeric(1))
  if (state$t == 0L) lk_new else lk_new - state$log_kernel
}

# ------------------------------------------------------------------
# Advance the inner SMC by a single tolerance 'epsilon'
# (one iteration of the outer t-loop of algorithm rare-event-smc:
#  reweight -> normalise -> resample[-if-degenerate] -> move).
#
# Returns the updated state.  The incremental contribution to the
# likelihood estimate, log( sum_n wtilde_n ), is added to state$log_lik
# and also returned in state$last_log_incr.
# ------------------------------------------------------------------
re_smc_step <- function(state, epsilon, model,
                        alpha         = 0.5,   # resample if ESS < alpha * Nu
                        resample_scheme = "multinomial",
                        move          = "pcn-mala",  # "pcn-mala" | "mh" | "gaussian"
                        adapt_nmoves  = TRUE,
                        c_move        = 0.2,
                        max_moves     = 100L) {
  Nu <- state$Nu

  ## --- reweight (lines 3-9) ----------------------------------------
  log_incr <- re_smc_log_incremental(state, epsilon, model)
  log_wtilde <- state$log_w + log_incr               # wtilde_n = w_{prev,n} * r_n
  last_log_incr <- log_sum_exp(log_wtilde)           # log sum_n wtilde_n

  state$log_lik       <- state$log_lik + last_log_incr
  state$last_log_incr <- last_log_incr

  ## --- normalise (line 10) -----------------------------------------
  state$log_w <- normalise_log_weights(log_wtilde)

  ## update cached kernel values to the new tolerance
  state$log_kernel <- vapply(state$x,
                             function(xx) model$log_abc_kernel(xx, epsilon),
                             numeric(1))

  ## --- resample if degenerate (lines 11-15) ------------------------
  ess <- ess_from_log_weights(state$log_w)
  if (ess < alpha * Nu) {
    anc <- resample(state$log_w, resample_scheme)
    state$u          <- state$u[anc]
    state$x          <- state$x[anc]
    state$log_kernel <- state$log_kernel[anc]
    state$log_w      <- rep(-log(Nu), Nu)
  }

  ## --- move (line 16) ----------------------------------------------
  state$last_epsilon <- epsilon
  state <- .re_move_all(state, epsilon, model, move,
                        adapt_nmoves, c_move, max_moves)

  state$t <- state$t + 1L
  state
}

# ------------------------------------------------------------------
# Stand-alone run of the rare event SMC over a *given* tolerance
# schedule (eps_1 > ... > eps_T).  Returns the final state, whose
# state$log_lik is the log of the ABC-likelihood estimate
#       lbar(y | theta) = prod_{t=1}^T sum_n wtilde^n_t
# (equation eq:rare_event_lld in the paper).
# ------------------------------------------------------------------
re_smc_run <- function(model, theta, Nu, epsilon_schedule,
                       alpha           = 0.5,
                       resample_scheme = "multinomial",
                       move            = "pcn-mala",
                       move_step0      = 1,
                       adapt_nmoves    = TRUE,
                       c_move          = 0.2,
                       max_moves       = 100L) {
  move  <- .resolve_move(move, model)
  state <- re_smc_init(model, theta, Nu, move_step0 = move_step0)
  for (eps in epsilon_schedule) {
    state <- re_smc_step(state, eps, model,
                         alpha = alpha, resample_scheme = resample_scheme,
                         move = move, adapt_nmoves = adapt_nmoves,
                         c_move = c_move, max_moves = max_moves)
  }
  state
}
