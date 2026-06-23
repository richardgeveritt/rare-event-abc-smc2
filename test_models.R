# =====================================================================
# test_models.R  --  smoke tests + reparameterisation checks for the
# MA(q) and Lotka-Volterra models, exercised through all three
# algorithms.  Run from the project root:
#
#     Rscript R/test_models.R
#
# These are sanity checks (the algorithms run; the reparameterisation
# matches the direct simulator; the rare-event likelihood matches a
# brute-force importance-sampling estimate; parameters move towards the
# truth), not a formal validation.  Settings are kept small so the
# whole script runs in a few minutes (the LV inner-SMC moves dominate);
# raise the particle counts / use a finer LV step_size for real use.
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/ma.R")
suppressMessages(source("R/lv.R"))

# weighted posterior mean of (vector) theta
wmean <- function(res) {
  Th <- do.call(rbind, lapply(res$theta, as.numeric))
  colSums(res$weights * Th)
}

# Unbiased-scale estimate of log l(y|theta) from the rare event SMC:
# log of the average of several independent likelihood estimates
# (averaging on the natural, not log, scale removes Jensen bias and so
#  is directly comparable to a low-variance importance-sampling value).
re_loglik_lme <- function(model, theta, Nu, schedule, reps = 6) {
  ll <- replicate(reps, re_smc_run(model, theta, Nu, schedule)$log_lik)
  log_sum_exp(ll) - log(reps)
}

# brute-force importance-sampling estimate of log l(y|theta) at tol eps
is_loglik <- function(model, theta, eps, M) {
  lk <- vapply(model$simulate(theta, M),
               function(x) model$log_abc_kernel(x, eps), numeric(1))
  log_sum_exp(lk) - log(M)
}

set.seed(1)

# =====================================================================
# MA(q) model
# =====================================================================
cat("#####################  MA(q)  #####################\n\n")

q_true     <- 2
theta_true <- c(0.6, 0.2)
n_obs      <- 30
model_ma   <- make_ma_model(rep(0, n_obs), q = q_true)        # placeholder
y_ma       <- model_ma$H(rnorm(n_obs + q_true), theta_true)   # data at truth
model_ma   <- make_ma_model(y_ma, q = q_true)

## ---- reparameterisation check: H vs direct simulator --------------
Xrep <- do.call(rbind, lapply(1:3000, function(i)
  model_ma$H(rnorm(n_obs + q_true), theta_true)))
Xdir <- do.call(rbind, model_ma$simulate(theta_true, 3000))
cat(sprintf("reparam: mean|Δcol-mean|=%.3f  var(H)=%.3f var(direct)=%.3f (theory %.3f)\n\n",
            mean(abs(colMeans(Xrep) - colMeans(Xdir))),
            mean(apply(Xrep, 2, var)), mean(apply(Xdir, 2, var)),
            1 + sum(theta_true^2)))

## ---- rare event SMC likelihood vs brute-force IS ------------------
cat("Rare event SMC vs importance sampling (log-lik at eps=1):\n")
sched_ma <- c(20, 11, 6, 3, 1)
for (th in list(theta_true, c(0.2, 0.0), c(-0.4, 0.3))) {
  ref <- is_loglik(model_ma, th, eps = 1, M = 8000)
  est <- re_loglik_lme(model_ma, th, Nu = 200, schedule = sched_ma, reps = 4)
  cat(sprintf("  theta=(% .2f,% .2f)  IS=%.3f  rare-event=%.3f\n",
              th[1], th[2], ref, est))
}
cat("\n")

## ---- ABC-SMC ------------------------------------------------------
fit_ma_abc <- abc_smc(model_ma, N_theta = 150, Nx = 20,
                      epsilon_final = 1, beta = 0.9)
cat(sprintf("ABC-SMC                 posterior mean: (% .3f, % .3f)   logZ=%.2f  iters=%d\n",
            wmean(fit_ma_abc)[1], wmean(fit_ma_abc)[2],
            fit_ma_abc$log_evidence, fit_ma_abc$n_iterations))

## ---- RE-ABC-SMC^2: compare the three inner u-moves ----------------
## The inner u-move matters.  We compare the generic MH (pCN) move with
## the gradient-based pCN-MALA and the exact Gaussian draw; the latter
## two are available because MA's H is linear, so the u-target is exactly
## Gaussian (grad_loglik_u / rtarget_gaussian are provided by ma.R).
## A FIXED shared schedule is used so the log-evidence estimates are
## directly comparable: a less-negative logZ means a lower-variance inner
## likelihood estimate (less downward SMC bias) -- the robust signal here.
## (Single-run posterior means are noisy at this Nu and only indicative.)
cat("RE-ABC-SMC^2 inner-move comparison (fixed schedule",
    paste(sched_ma, collapse = ","), "):\n")
for (mv in c("mh", "pcn-mala", "gaussian")) {
  fit <- re_abc_smc2(model_ma, N_theta = 150, Nu = 20,
                     epsilon_schedule = sched_ma, move = mv)
  cat(sprintf("  [%-8s]  posterior mean: (% .3f, % .3f)   logZ=%.2f\n",
              mv, wmean(fit)[1], wmean(fit)[2], fit$log_evidence))
}
cat(sprintf("(true theta = (% .3f, % .3f); expect logZ: mh < pcn-mala <= gaussian,\n",
            theta_true[1], theta_true[2]))
cat(" the better u-moves giving a lower-variance inner likelihood.)\n\n")

# =====================================================================
# Lotka-Volterra model  (coarse Euler step for a fast smoke test)
# =====================================================================
cat("################  Lotka-Volterra  ################\n\n")

theta_lv_true <- c(1, 0.005, 0.6)
step_lv  <- 0.05                                   # dim(u) = 15*40*3 = 1800
model_lv <- make_lv_model(step_size = step_lv)
cat(sprintf("dim(u) = %d  (n_fine=%d, intervals=%d)\n",
            model_lv$.du, model_lv$.n_fine, model_lv$.num_intervals))
y_lv     <- model_lv$H(rnorm(model_lv$.du), theta_lv_true)  # data at truth
model_lv <- make_lv_model(y = y_lv, step_size = step_lv)

## ---- reparameterisation check: H is finite & on the data scale ----
Hs <- lapply(1:200, function(i) model_lv$H(rnorm(model_lv$.du), theta_lv_true))
prey_mean <- Reduce(`+`, Hs)[, 1] / length(Hs)
cat(sprintf("reparam: any non-finite = %s   mean-prey range = [%.0f, %.0f]\n\n",
            any(!is.finite(unlist(Hs))), min(prey_mean), max(prey_mean)))

## ---- rare event SMC likelihood vs brute-force IS ------------------
cat("Rare event SMC vs importance sampling (log-lik at eps=80):\n")
sched_lv <- c(350, 180, 100, 80)
for (th in list(theta_lv_true, c(1.4, 0.005, 0.6))) {
  ref <- is_loglik(model_lv, th, eps = 80, M = 800)
  est <- re_loglik_lme(model_lv, th, Nu = 80, schedule = sched_lv, reps = 3)
  cat(sprintf("  theta=(% .3f, %.4f, %.3f)  IS=%.3f  rare-event=%.3f\n",
              th[1], th[2], th[3], ref, est))
}
cat("\n")

## ---- ABC-SMC and RE-ABC-SMC^2 (short fixed schedule) --------------
## NB: LV inference (3 parameters over orders of magnitude, 32-dim raw
## data) is hard; at these smoke-test sizes the posteriors are rough.
## The validated pieces are the reparameterisation and the likelihood
## check above; raise N_theta / Nu / max_moves for a sharp posterior.
fit_lv_abc <- abc_smc(model_lv, N_theta = 70, Nx = 6,
                      epsilon_schedule = sched_lv)
fit_lv_re2 <- re_abc_smc2(model_lv, N_theta = 25, Nu = 30,
                          epsilon_schedule = sched_lv, max_moves = 4)
cat(sprintf("ABC-SMC      posterior mean: (% .3f, %.4f, %.3f)  logZ=%.2f iters=%d\n",
            wmean(fit_lv_abc)[1], wmean(fit_lv_abc)[2], wmean(fit_lv_abc)[3],
            fit_lv_abc$log_evidence, fit_lv_abc$n_iterations))
cat(sprintf("RE-ABC-SMC^2 posterior mean: (% .3f, %.4f, %.3f)  logZ=%.2f iters=%d\n",
            wmean(fit_lv_re2)[1], wmean(fit_lv_re2)[2], wmean(fit_lv_re2)[3],
            fit_lv_re2$log_evidence, fit_lv_re2$n_iterations))
cat(sprintf("(true theta = (% .3f, %.4f, %.3f))\n",
            theta_lv_true[1], theta_lv_true[2], theta_lv_true[3]))
