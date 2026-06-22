# =====================================================================
# lv.R  --  Lotka-Volterra (stochastic, chemical Langevin) model in the
#           standard model format (see example_model_gaussian.R /
#           R/README.md).
#
# The model is the smfsb LV predator-prey reaction network simulated
# with the chemical Langevin equation (CLE) via Euler-Maruyama:
#       x_{k+1} = x_k + S ( h(x_k) dt + sqrt(h(x_k)) dw_k ),
#       dw_k = sqrt(dt) z_k,   z_k ~ N(0, I_v),
# where S is the stoichiometry matrix, h the hazards and v the number
# of reactions (= 3 for LV).
#
# Reparameterisation
# ------------------
# All randomness is in the Brownian increments.  Collecting every
# per-step standard normal z_k (over all fine Euler steps of all
# observation intervals) into one vector u gives
#       x = H(u, theta),    phi(u | theta) = N(0, I)   (theta-free).
# The MCMC move on u is a preconditioned Crank-Nicolson (pCN) proposal.
#
# Requires the 'smfsb' package for the LV data (LVperfect) and the
# reaction network (stoichiometry).
# =====================================================================

require(smfsb)

# ---------------------------------------------------------------------
# Deterministic CLE integration of one observation interval, consuming
# a pre-drawn (n_fine x v) matrix Z of standard normals.  Negative
# states are reflected back to positive (x <- |x|), matching smfsb's
# StepCLE; this keeps the populations positive and the diffusion
# well-defined (hazards stay non-negative, so sqrt(h) is real).
# ---------------------------------------------------------------------
lv_cle_interval <- function(x, theta, n_fine, dt, S, Z, cap = 1e8) {
  sdt <- sqrt(dt)
  for (k in seq_len(n_fine)) {
    h  <- c(theta[1] * x[1],
            theta[2] * x[1] * x[2],
            theta[3] * x[2])
    dx <- S %*% (h * dt + sqrt(pmax(h, 0)) * (sdt * Z[k, ]))
    x  <- x + as.vector(dx)
    x[!is.finite(x)] <- cap                  # guard Euler overflow / blow-up
    x[x < 0] <- -x[x < 0]                    # reflecting boundary at 0
    x <- pmin(x, cap)
  }
  x
}

# =====================================================================
# Model constructor.
#
#   y          observed series, a (num_obs x 2) matrix (default LVperfect)
#   start_time, end_time, time_step   observation grid
#   step_size  fine Euler-Maruyama step (<= time_step); smaller is more
#              accurate but increases dim(u) = num_intervals * n_fine * v
#   x0         initial state (prey, predator)
#   pcn_rho    step size of the pCN move on u, in (0, 1]
#   rw_sd_theta s.d. of the (fallback) Gaussian random-walk on theta
#               (used only when adapt_theta_proposal = FALSE)
# =====================================================================
make_lv_model <- function(y          = NULL,
                          start_time  = 0,
                          end_time    = 30,
                          time_step   = 2,
                          step_size   = 0.0005,
                          x0          = c(50, 100),
                          pcn_rho     = 0.02,
                          rw_sd_theta = 0.05) {

  if (is.null(y)) { data(LVdata); y <- LVperfect }
  data(spnModels)
  y_mat <- as.matrix(y)
  S     <- t(LV$Post - LV$Pre)              # 2 x 3 stoichiometry
  v     <- ncol(S)                          # number of reactions (3)

  num_intervals <- round((end_time - start_time) / time_step)
  n_fine        <- round(time_step / step_size)
  du            <- num_intervals * n_fine * v   # dimension of u

  # x = H(u, theta): run the CLE deterministically using the normals u.
  transform <- function(u, theta) {
    Z   <- matrix(u, ncol = v, byrow = TRUE)    # (num_intervals*n_fine) x v
    x   <- x0
    out <- matrix(0, num_intervals + 1L, 2L)
    out[1, ] <- x
    for (j in seq_len(num_intervals)) {
      rows <- ((j - 1L) * n_fine + 1L):(j * n_fine)
      x <- lv_cle_interval(x, theta, n_fine, step_size, S, Z[rows, , drop = FALSE])
      out[j + 1L, ] <- x
    }
    out
  }

  distance <- function(x) sqrt(sum((as.vector(x) - as.vector(y_mat))^2))
  s_rho    <- sqrt(1 - pcn_rho^2)

  list(
    y = y_mat,

    # ---- theta level ------------------------------------------------
    # Prior: log(theta_i) ~ U(-6, 2) independently (i = 1, 2, 3).
    rprior = function(N) {
      M <- exp(matrix(runif(3 * N, -6, 2), nrow = N, ncol = 3))
      lapply(seq_len(N), function(m) M[m, ])
    },
    dprior = function(theta) {
      if (any(theta <= 0)) return(-Inf)
      lt <- log(theta)
      if (any(lt < -6) || any(lt > 2)) return(-Inf)
      -sum(lt)                                  # density up to a constant
    },
    rproposal = function(theta) theta + rnorm(3, 0, rw_sd_theta),
    dproposal = function(to, from) sum(dnorm(to, from, rw_sd_theta, log = TRUE)),

    # ---- direct simulation (for ABC-SMC) ----------------------------
    simulate = function(theta, Nx) {
      lapply(seq_len(Nx), function(i) transform(rnorm(du), theta))
    },

    # ---- reparameterised pieces (rare event SMC / RE-ABC-SMC^2) ------
    rphi = function(theta, Nu) lapply(seq_len(Nu), function(i) rnorm(du)),
    dphi = function(u, theta) sum(dnorm(u, log = TRUE)),
    H    = function(u, theta) transform(u, theta),

    # pCN MCMC move on u (Gaussian reference measure)
    ru_move = function(u, theta) s_rho * u + pcn_rho * rnorm(length(u)),
    du_move = function(to, from, theta)
      sum(dnorm(to, mean = s_rho * from, sd = pcn_rho, log = TRUE)),

    # ---- ABC kernel: Gaussian on the raw-series Euclidean distance ---
    log_abc_kernel = function(x, epsilon) -0.5 * (distance(x) / epsilon)^2,

    # exposed for testing / inspection
    .du = du, .n_fine = n_fine, .num_intervals = num_intervals
  )
}
