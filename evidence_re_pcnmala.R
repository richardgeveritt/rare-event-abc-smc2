# =====================================================================
# evidence_re_pcnmala.R
#
# Re-run of the RE-ABC-SMC^2 log-evidence experiment of
# evidence_model_selection.R, but with the inner u-move set to MALA-pCN
# (move = "pcn-mala") instead of the exact Gaussian draw.  pcn-mala is
# an MCMC kernel (not an exact sampler), so its rare-event likelihood
# has a little more variance than the gaussian move -- this checks how
# much the (default) gradient move costs in evidence accuracy.
#
# Same data sets (same seed, n = 100), same fitted models, adaptive
# tolerance to eps = 1.  Prints the 3x3 log-evidence table.
#
#   Rscript R/evidence_re_pcnmala.R
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

# ---- identical data-generating thetas and data sets as the main run --
true_theta <- lapply(orders, function(q) {
  repeat {
    th <- as.numeric(simulate_prior_maq(1, q))
    if (abs(th[q]) > 0.3) return(th)
  }
})
datasets <- lapply(seq_along(orders), function(i) {
  q <- orders[i]
  make_ma_model(rep(0, n_obs), q)$H(rnorm(n_obs + q), true_theta[[i]])
})
lab <- paste0("MA(", orders, ")")

cat(sprintf("RE-ABC-SMC^2 with move = pcn-mala  (n=%d, N_theta=%d, adaptive to eps=%g)\n\n",
            n_obs, N_theta, eps_fin))

logZ <- matrix(NA_real_, 3, 3, dimnames = list(data = lab, fit = lab))
eps_reached <- logZ
for (di in seq_along(orders)) {
  y <- datasets[[di]]
  for (fi in seq_along(orders)) {
    model <- make_ma_model(y, orders[fi])
    set.seed(1000 * di + fi)          # same seed as the gaussian/ABC run
    fr <- re_abc_smc2(model, N_theta = N_theta, Nu = 40,
                      epsilon_final = eps_fin, beta = 0.9,
                      move = "pcn-mala", max_steps = 150L, max_moves = 20L)
    logZ[di, fi] <- fr$log_evidence
    eps_reached[di, fi] <- tail(fr$epsilon, 1)
    cat(sprintf("  data %-7s fit %-7s | RE-ABC-SMC^2 (pcn-mala) logZ=%8.2f (eps %.2f, %2d it)\n",
                lab[di], lab[fi], fr$log_evidence, tail(fr$epsilon, 1), fr$n_iterations))
  }
}

cat("\n== RE-ABC-SMC^2 (pcn-mala) log-evidence (rows = data, cols = fit) ==\n")
print(round(logZ, 2))
sel <- lab[apply(logZ, 1, which.max)]
cat("selected (argmax over fitted model):",
    paste(sprintf("%s->%s", lab, sel), collapse = "  "), "\n")
cat("\nFinal tolerance reached:\n  ",
    paste(sprintf("%.2f", as.vector(t(eps_reached))), collapse = " "), "\n")
