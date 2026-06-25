# =====================================================================
# ma.R  --  Moving-average MA(q) model in the standard model format
#           (see example_model_gaussian.R / R/README.md).
#
# Reparameterisation
# ------------------
# An MA(q) series is
#       y_k = u_k + theta_1 u_{k-1} + ... + theta_q u_{k-q},
# where the innovations u are i.i.d. N(0,1).  Collecting all n+q
# innovations into the vector u gives the deterministic transform
#       x = H(u, theta) = filter(u, c(1, theta))
# with reference distribution  phi(u | theta) = N(0, I_{n+q})  (which
# does not depend on theta).  The MCMC move on u is a preconditioned
# Crank-Nicolson (pCN) proposal, the natural choice for a Gaussian
# reference measure (its prior/proposal terms cancel, so the MH ratio
# reduces to the ABC-kernel ratio).
# =====================================================================

# ---------------------------------------------------------------------
# Helpers for the (exact, normalised) uniform prior on the MA(q)
# identifiability region {theta : roots of 1 + theta_1 x + ... lie
# strictly outside the unit circle}.
# ---------------------------------------------------------------------

# Exact volume of the MA(q) identifiability region.
calculate_maq_volume <- function(q) {
  if (q == 0) return(1)
  volume <- 2                                  # base volume for MA(1)
  if (q == 1) return(volume)
  for (k in 2:q) {                             # Durbin-Levinson Jacobian
    a <- ceiling((k - 1) / 2)
    b <- floor((k - 1) / 2)
    I_k <- (2^k * factorial(a) * factorial(b)) / factorial(k)
    volume <- volume * I_k
  }
  volume
}

# Log-density of the normalised uniform prior at theta.
log_density_prior_normalized <- function(theta) {
  q <- length(theta)
  roots <- polyroot(c(1, theta))               # 1 + theta_1 x + ... + theta_q x^q
  if (all(Mod(roots) > 1)) -log(calculate_maq_volume(q)) else -Inf
}

# Rejection sampler for the uniform prior over the identifiability
# region (theta_i is bounded by choose(q, i)).
simulate_prior_maq <- function(n_samples, q) {
  bounds  <- vapply(1:q, function(i) choose(q, i), numeric(1))
  samples <- matrix(NA_real_, nrow = n_samples, ncol = q)
  count <- 1L
  while (count <= n_samples) {
    proposal <- runif(q, min = -bounds, max = bounds)
    if (all(Mod(polyroot(c(1, proposal))) > 1)) {
      samples[count, ] <- proposal
      count <- count + 1L
    }
  }
  samples
}

# ---------------------------------------------------------------------
# Deterministic transform  x = H(u, theta).
# u is a length-(n+q) vector of innovations; returns the n observed
# points (the first q filtered values, which lack history, are dropped).
# ---------------------------------------------------------------------
ma_transform <- function(u, theta) {
  q  <- length(theta)
  yf <- stats::filter(u, filter = c(1, theta), method = "convolution",
                      sides = 1)
  as.numeric(yf[(q + 1):length(u)])
}

# ---------------------------------------------------------------------
# Adjoint of the linear map  u -> x = A u  ( = ma_transform ): given a
# length-n residual r, returns A' r (length n+q), matrix-free.  Used for
# the likelihood gradient.
# ---------------------------------------------------------------------
ma_transform_adjoint <- function(r, theta) {
  q <- length(theta); f <- c(1, theta); n <- length(r); d <- n + q
  out <- numeric(d)
  for (m in seq_len(q + 1)) {                  # (A'r)[j] = sum_m f[m] r[j-q+m-1]
    idx <- seq_len(d) - q + m - 1
    ok  <- idx >= 1 & idx <= n
    out[ok] <- out[ok] + f[m] * r[idx[ok]]
  }
  out
}

# =====================================================================
# Model constructor.
#
#   y          observed length-n series
#   q          MA order (= length(theta));  must be >= 1
#   pcn_rho    step size of the pCN move on u, in (0, 1]
#   rw_sd_theta s.d. of the (fallback) Gaussian random-walk on theta,
#               used only when adapt_theta_proposal = FALSE
# =====================================================================
make_ma_model <- function(y, q, pcn_rho = 0.1, rw_sd_theta = 0.1) {
  n  <- length(y)
  du <- n + q                                   # dimension of u
  y  <- as.numeric(y)

  distance <- function(x) sqrt(sum((x - y)^2))
  s_rho    <- sqrt(1 - pcn_rho^2)               # pCN retention factor

  # cache of A'A and A'y for the most recent theta (A = Jacobian of H,
  # which depends on theta only, not epsilon); speeds up rtarget_gaussian
  # a lot when called repeatedly for the same theta (e.g. the inner-SMC
  # reruns inside the RE-ABC-SMC^2 external move).
  gcache <- new.env(parent = emptyenv())
  gcache$theta <- NULL

  list(
    y = y,

    # ---- theta level ------------------------------------------------
    rprior = function(N) {
      M <- simulate_prior_maq(N, q)
      lapply(seq_len(N), function(m) M[m, ])
    },
    dprior    = function(theta) log_density_prior_normalized(theta),
    rproposal = function(theta) theta + rnorm(q, 0, rw_sd_theta),
    dproposal = function(to, from) sum(dnorm(to, from, rw_sd_theta, log = TRUE)),

    # ---- direct simulation (for ABC-SMC) ----------------------------
    simulate = function(theta, Nx) {
      lapply(seq_len(Nx), function(i) ma_transform(rnorm(du), theta))
    },

    # ---- reparameterised pieces (rare event SMC / RE-ABC-SMC^2) ------
    rphi = function(theta, Nu) lapply(seq_len(Nu), function(i) rnorm(du)),
    dphi = function(u, theta) sum(dnorm(u, log = TRUE)),
    H    = function(u, theta) ma_transform(u, theta),

    # pCN MCMC move on u (Gaussian reference measure)
    ru_move = function(u, theta) s_rho * u + pcn_rho * rnorm(length(u)),
    du_move = function(to, from, theta)
      sum(dnorm(to, mean = s_rho * from, sd = pcn_rho, log = TRUE)),

    # ---- ABC kernel: Gaussian on the raw-series Euclidean distance ---
    log_abc_kernel = function(x, epsilon) -0.5 * (distance(x) / epsilon)^2,

    # ---- gradient of log P_eps(y | H(u,theta)) w.r.t. u (for pCN-MALA) -
    #   = -(1/eps^2) A'(A u - y)
    grad_loglik_u = function(u, theta, epsilon)
      -ma_transform_adjoint(ma_transform(u, theta) - y, theta) / epsilon^2,

    # ---- exact independent draws from the Gaussian u-target -----------
    #   pi(u) ∝ P_eps(y|H(u,theta)) phi(u) = N(m, C),
    #   C = (I + A'A/eps^2)^{-1},  m = C A'y / eps^2  (H linear, Gaussian kernel)
    rtarget_gaussian = function(theta, epsilon, n_draws) {
      d <- n + length(theta)
      if (is.null(gcache$theta) || length(theta) != length(gcache$theta) ||
          any(theta != gcache$theta)) {            # rebuild A'A, A'y for new theta
        A <- vapply(seq_len(d), function(j) {
          e <- numeric(d); e[j] <- 1; ma_transform(e, theta)
        }, numeric(n))
        gcache$theta <- theta
        gcache$AtA   <- crossprod(A)
        gcache$Aty   <- as.vector(crossprod(A, y))
      }
      P <- diag(d) + gcache$AtA / epsilon^2              # precision
      m <- as.vector(solve(P, gcache$Aty / epsilon^2))
      R <- chol(P)                                       # P = R'R
      lapply(seq_len(n_draws), function(i) m + backsolve(R, rnorm(d)))
    }
  )
}
