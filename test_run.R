# =====================================================================
# test_run.R  --  quick sanity check that all three algorithms run.
#
#   Rscript R/test_run.R     (run from the project root)
#
# Uses the throw-away Gaussian example model.  This is NOT a validation
# of correctness, just a smoke test that the code executes and returns
# sensible (sigma ~ 3) posterior means.
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/example_model_gaussian.R")

set.seed(1)

## ---- simulate observed data --------------------------------------
d_true     <- 10
sigma_true <- 3
y     <- sigma_true * abs(rnorm(d_true))
model <- make_gaussian_model(y)

wmean <- function(res) sum(res$weights * unlist(res$theta))

## ---- 1. Rare event SMC: ABC-likelihood estimate for a fixed theta -
cat("== Rare event SMC (likelihood estimate at theta = 3) ==\n")
sched <- c(20, 12, 7, 4, 2.5, 1.5)
#re <- re_smc_run(model, theta = 3, Nu = 200, epsilon_schedule = sched)
#cat(sprintf("  log-likelihood estimate: %.3f\n\n", re$log_lik))

## ---- 2. ABC-SMC (comparator) -------------------------------------
cat("== ABC-SMC ==\n")
fit_abc <- abc_smc(model, N_theta = 200, Nx = 20,
                   epsilon_final = 1.5, beta = 0.9, verbose = TRUE)
cat(sprintf("  posterior mean(sigma): %.3f   log-evidence: %.3f   iters: %d\n\n",
            wmean(fit_abc), fit_abc$log_evidence, fit_abc$n_iterations))

## ---- 3. RE-ABC-SMC^2 ---------------------------------------------
cat("== RE-ABC-SMC^2 ==\n")
fit_re2 <- re_abc_smc2(model, N_theta = 100, Nu = 60,
                       epsilon_final = 1.5, beta = 0.9, verbose = TRUE)
cat(sprintf("  posterior mean(sigma): %.3f   log-evidence: %.3f   iters: %d\n",
            wmean(fit_re2), fit_re2$log_evidence, fit_re2$n_iterations))
