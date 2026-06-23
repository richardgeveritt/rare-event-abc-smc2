# =====================================================================
# evidence_model_selection.R
#
# Check the log-evidence (log marginal likelihood) reported by ABC-SMC
# and RE-ABC-SMC^2 as a model-selection tool.
#
# We simulate one data set from each of MA(2), MA(3), MA(4), then fit
# every model to every data set with both algorithms, using an ADAPTIVE
# tolerance with final value eps = 1.  Because both algorithms use the
# same ABC kernel and the same final tolerance, their SMC evidence
# estimates target the same ABC marginal likelihood log Z(eps=1), which
# telescopes along the (adaptive) schedule and so is comparable across
# models and across algorithms.
#
# Output: a 3x3 table of log-evidence for each algorithm (rows = the
# model that generated the data, columns = the fitted model).
#
# RE-ABC-SMC^2 uses move = "gaussian" (exact draws from the Gaussian
# u-target; the lowest-variance evidence estimator for the linear MA
# model).
#
#   Rscript R/evidence_model_selection.R
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/ma.R")

set.seed(123)

orders  <- c(2, 3, 4)
n_obs   <- 100
N_theta <- 500
eps_fin <- 1

# ---- a "true" theta for each order, with a non-negligible top
#      coefficient so the data genuinely has that order -----------------
true_theta <- lapply(orders, function(q) {
  repeat {
    th <- as.numeric(simulate_prior_maq(1, q))
    if (abs(th[q]) > 0.3) return(th)
  }
})
names(true_theta) <- paste0("MA(", orders, ")")

# ---- one data set per generating model (same length n_obs) -----------
datasets <- lapply(seq_along(orders), function(i) {
  q <- orders[i]
  make_ma_model(rep(0, n_obs), q)$H(rnorm(n_obs + q), true_theta[[i]])
})
names(datasets) <- paste0("MA(", orders, ")")

cat("True coefficients of the data-generating models:\n")
for (nm in names(true_theta))
  cat(sprintf("  %-7s theta = (%s)\n", nm,
              paste(sprintf("% .2f", true_theta[[nm]]), collapse = ", ")))
cat(sprintf("\nFitting every model to every data set (n=%d, N_theta=%d, adaptive to eps=%g).\n\n",
            n_obs, N_theta, eps_fin))

lab <- paste0("MA(", orders, ")")
logZ_abc <- matrix(NA_real_, 3, 3, dimnames = list(data = lab, fit = lab))
logZ_re2 <- matrix(NA_real_, 3, 3, dimnames = list(data = lab, fit = lab))
eps_abc  <- logZ_abc; eps_re2 <- logZ_re2

for (di in seq_along(orders)) {
  y <- datasets[[di]]
  for (fi in seq_along(orders)) {
    qf <- orders[fi]
    model <- make_ma_model(y, qf)

    set.seed(1000 * di + fi)
    fa <- abc_smc(model, N_theta = N_theta, Nx = 25,
                  epsilon_final = eps_fin, beta = 0.9,
                  max_steps = 150L, max_moves = 10L)
    set.seed(1000 * di + fi)
    fr <- re_abc_smc2(model, N_theta = N_theta, Nu = 40,
                      epsilon_final = eps_fin, beta = 0.9,
                      move = "gaussian", max_steps = 150L)

    logZ_abc[di, fi] <- fa$log_evidence; eps_abc[di, fi] <- tail(fa$epsilon, 1)
    logZ_re2[di, fi] <- fr$log_evidence; eps_re2[di, fi] <- tail(fr$epsilon, 1)
    cat(sprintf("  data %-7s fit %-7s | ABC-SMC logZ=%8.2f (eps %.2f, %2d it)  |  RE-ABC-SMC^2 logZ=%8.2f (eps %.2f, %2d it)\n",
                lab[di], lab[fi], fa$log_evidence, tail(fa$epsilon, 1), fa$n_iterations,
                fr$log_evidence, tail(fr$epsilon, 1), fr$n_iterations))
  }
}

show_table <- function(M, title) {
  cat("\n==", title, "(rows = data-generating model, cols = fitted model) ==\n")
  print(round(M, 2))
  sel <- lab[apply(M, 1, which.max)]
  cat("selected (argmax over fitted model):",
      paste(sprintf("%s->%s", lab, sel), collapse = "  "), "\n")
}
show_table(logZ_abc, "ABC-SMC      log-evidence")
show_table(logZ_re2, "RE-ABC-SMC^2 log-evidence")

cat("\nFinal tolerance reached (should all be ~", eps_fin, "for comparability):\n", sep = "")
cat("  ABC-SMC :", paste(sprintf("%.2f", as.vector(t(eps_abc))), collapse = " "), "\n")
cat("  RE2     :", paste(sprintf("%.2f", as.vector(t(eps_re2))), collapse = " "), "\n")
