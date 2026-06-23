# =====================================================================
# converge_to_exact.R
#
# Convergence check: with a Gaussian ABC kernel on the RAW data the ABC
# posterior tends to the EXACT posterior as eps -> 0.  We therefore run
# both ABC algorithms ADAPTIVELY down to eps = 0.1 and compare the final
# theta-posteriors with the exact MA(2) posterior (ma_exact_posterior.R).
#
#   * RE-ABC-SMC^2 uses move = "gaussian": for the linear MA model the
#     u-target is exactly Gaussian, so this draws the u-particles exactly
#     -- it reaches eps = 0.1 cheaply and should match the exact posterior.
#   * ABC-SMC uses the plain Nx-point IS likelihood; at eps = 0.1 on
#     30-dimensional raw data this is essentially degenerate (a random
#     simulation is never within 0.1 of y), so it will stall/collapse
#     well above 0.1.  Run for completeness; not a fair comparison.
#
#   Rscript R/converge_to_exact.R
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/ma.R")
source("R/ma_exact_posterior.R")
suppressMessages({ library(ggplot2) })

set.seed(1)

# ---- MA(2) model and data (same setup as the evolution figure) -----
q_true     <- 2
theta_true <- c(0.6, 0.2)
n_obs      <- 30
m0    <- make_ma_model(rep(0, n_obs), q = q_true)
y_ma  <- m0$H(rnorm(n_obs + q_true), theta_true)
model <- make_ma_model(y_ma, q = q_true)

eps_target <- 0.1
N_theta    <- 1000

# ---- exact posterior (ground truth) --------------------------------
cat("Sampling the exact posterior by MCMC ...\n")
exact <- ma_exact_mcmc(y_ma, q_true, n_iter = 80000, burn_in = 20000)
ex    <- exact$samples
cat(sprintf("  exact: mean=(%.3f, %.3f)  sd=(%.3f, %.3f)  cor=%.3f  (acc %.2f)\n",
            mean(ex[, 1]), mean(ex[, 2]), sd(ex[, 1]), sd(ex[, 2]),
            cor(ex[, 1], ex[, 2]), exact$acc_rate))

# ---- RE-ABC-SMC^2 with the exact Gaussian u-move, adaptive to 0.1 ---
cat("Running RE-ABC-SMC^2 (move = gaussian) adaptively to eps =", eps_target, "...\n")
fit_re2 <- re_abc_smc2(model, N_theta = N_theta, Nu = 50,
                       epsilon_final = eps_target, beta = 0.9,
                       move = "gaussian", record_history = TRUE)
cat(sprintf("  reached eps = %.3f in %d iterations\n",
            tail(fit_re2$epsilon, 1), fit_re2$n_iterations))

# ---- ABC-SMC, adaptive to 0.1 (best effort) ------------------------
cat("Running ABC-SMC adaptively to eps =", eps_target, "(may stall) ...\n")
fit_abc <- abc_smc(model, N_theta = N_theta, Nx = 30,
                   epsilon_final = eps_target, beta = 0.9,
                   max_steps = 60L, max_moves = 15L, record_history = TRUE)
cat(sprintf("  reached eps = %.3f in %d iterations\n",
            tail(fit_abc$epsilon, 1), fit_abc$n_iterations))

# ---- weighted summaries of the final theta-populations -------------
wsummary <- function(fit) {
  Th <- do.call(rbind, fit$theta); w <- fit$weights / sum(fit$weights)
  mu <- colSums(w * Th)
  v  <- colSums(w * sweep(Th, 2, mu)^2)
  cv <- sum(w * (Th[, 1] - mu[1]) * (Th[, 2] - mu[2]))
  ess <- 1 / sum(w^2)
  list(mean = mu, sd = sqrt(v), cor = cv / prod(sqrt(v)), ess = ess)
}
s_re2 <- wsummary(fit_re2); s_abc <- wsummary(fit_abc)

cat("\n                 mean(theta1, theta2)     sd(theta1, theta2)    cor     ESS\n")
fmt <- function(tag, s, eps)
  cat(sprintf("  %-16s (% .3f, % .3f)   (%.3f, %.3f)   % .3f   %s\n",
              sprintf("%s eps=%.2f", tag, eps), s$mean[1], s$mean[2],
              s$sd[1], s$sd[2], s$cor,
              if (is.null(s$ess)) "" else sprintf("%.0f", s$ess)))
cat(sprintf("  %-16s (% .3f, % .3f)   (%.3f, %.3f)   % .3f\n",
            "EXACT", mean(ex[,1]), mean(ex[,2]), sd(ex[,1]), sd(ex[,2]),
            cor(ex[,1], ex[,2])))
fmt("RE-ABC-SMC^2", s_re2, tail(fit_re2$epsilon, 1))
fmt("ABC-SMC",      s_abc, tail(fit_abc$epsilon, 1))

# =====================================================================
# Figure: final posteriors vs exact
# =====================================================================
cols <- c("Exact" = "black", "RE-ABC-SMC^2 (gaussian)" = "#1f77b4",
          "ABC-SMC" = "#d62728")
resample <- function(fit, n = 4000) {
  Th <- do.call(rbind, fit$theta)
  j  <- sample.int(nrow(Th), n, replace = TRUE, prob = fit$weights)
  Th[j, ] + matrix(rnorm(2 * n, 0, 0.01), n, 2)        # tiny jitter for display
}
re2_s <- resample(fit_re2); abc_s <- resample(fit_abc)
marg <- rbind(
  data.frame(theta1 = ex[,1],    theta2 = ex[,2],    Method = "Exact"),
  data.frame(theta1 = re2_s[,1], theta2 = re2_s[,2], Method = "RE-ABC-SMC^2 (gaussian)"),
  data.frame(theta1 = abc_s[,1], theta2 = abc_s[,2], Method = "ABC-SMC"))
marg$Method <- factor(marg$Method, levels = names(cols))

pdf("R/converge_to_exact.pdf", width = 10, height = 4)
# marginals
for (d in 1:2) {
  v <- if (d == 1) "theta1" else "theta2"
  g <- ggplot(marg, aes(.data[[v]], colour = Method, linetype = Method)) +
    geom_density(linewidth = 0.9) +
    geom_vline(xintercept = theta_true[d], colour = "grey60", linetype = 3) +
    scale_colour_manual(values = cols) +
    scale_linetype_manual(values = c("Exact" = 1, "RE-ABC-SMC^2 (gaussian)" = 1,
                                     "ABC-SMC" = 2)) +
    labs(title = bquote("Final posterior on " * theta[.(d)] *
                        " (adaptive to eps=0.1) vs exact"),
         x = bquote(theta[.(d)]), y = "density") +
    theme_bw(base_size = 11) + theme(legend.position = "top")
  print(g)
}
# joint
grid <- ma_exact_grid_posterior(y_ma)
gj <- ggplot() +
  geom_density_2d(data = data.frame(theta1 = ex[,1], theta2 = ex[,2]),
                  aes(theta1, theta2), colour = "black", bins = 8, linewidth = 0.3) +
  geom_point(data = subset(marg, Method != "Exact"),
             aes(theta1, theta2, colour = Method), alpha = 0.15, size = 0.6) +
  geom_point(aes(theta_true[1], theta_true[2]), shape = 4, size = 3,
             stroke = 1.2, colour = "grey20") +
  scale_colour_manual(values = cols) +
  coord_cartesian(xlim = c(0, 1.4), ylim = c(-0.4, 0.8)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 2))) +
  labs(title = "Joint posterior vs exact (black contours; x = true theta)",
       x = expression(theta[1]), y = expression(theta[2])) +
  theme_bw(base_size = 11) + theme(legend.position = "top")
print(gj)
invisible(dev.off())
cat("\nWrote R/converge_to_exact.pdf\n")
