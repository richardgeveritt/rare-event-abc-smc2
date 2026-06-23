# =====================================================================
# visualise_theta_evolution.R
#
# Visualise how the posterior on theta evolves over the iterations
# (decreasing tolerances) of ABC-SMC and RE-ABC-SMC^2 on the MA(2)
# example, using a FIXED tolerance schedule and 1000 theta-particles,
# and compare against the EXACT (non-ABC) posterior sampled by MCMC
# (ma_exact_posterior.R) -- the ground truth.
#
# Both algorithms record the weighted theta-population at each target
# (record_history = TRUE); these are turned into the tidy format used by
# the ggsmc package (matrix2tidy).  Outputs:
#   * theta_evolution.pdf : weighted marginal posteriors at each
#     tolerance (ABC-SMC vs RE-ABC-SMC^2), with the exact posterior
#     overlaid as a black reference; plus a joint (theta_1,theta_2) page
#   * ggsmc animations of the marginal density and joint scatter -> GIFs
#
#   Rscript R/visualise_theta_evolution.R
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/ma.R")
source("R/ma_exact_posterior.R")
suppressMessages({ library(ggsmc); library(ggplot2) })

set.seed(1)

# ---- MA(2) model, data simulated at the true theta -----------------
q_true     <- 2
theta_true <- c(0.6, 0.2)
n_obs      <- 30
m0    <- make_ma_model(rep(0, n_obs), q = q_true)
y_ma  <- m0$H(rnorm(n_obs + q_true), theta_true)
model <- make_ma_model(y_ma, q = q_true)

# ---- fixed tolerance schedule and particle count -------------------
# NB: with move = "pcn-mala" and N_theta = 1000 the RE-ABC-SMC^2 run is
# the bottleneck (~15-20 min: the external theta-move reruns the inner
# SMC for every proposal).  To speed it up, lower re2_Nu / re2_max_moves
# below, shorten the schedule, or use move = "gaussian" (exact, cheap).
schedule    <- c(25, 16, 10, 6.5, 4, 2.6, 1.6, 1)
N_theta     <- 1000
re2_move    <- "pcn-mala"                 # inner u-move (now the default)
re2_Nu      <- 40                          # inner particles for RE-ABC-SMC^2
re2_max_moves <- 100L                      # cap on adaptive MCMC sweeps

cat(sprintf("MA(2): n_obs=%d, N_theta=%d, fixed schedule = %s\n",
            n_obs, N_theta, paste(schedule, collapse = ", ")))

cat("Running ABC-SMC ...\n")
fit_abc <- abc_smc(model, N_theta = N_theta, Nx = 30,
                   epsilon_schedule = schedule, record_history = TRUE)
cat("Running RE-ABC-SMC^2 (move = ", re2_move, ") ...\n", sep = "")
fit_re2 <- re_abc_smc2(model, N_theta = N_theta, Nu = re2_Nu,
                       epsilon_schedule = schedule, move = re2_move,
                       max_moves = re2_max_moves, record_history = TRUE)
cat("Sampling the EXACT posterior by MCMC ...\n")
exact <- ma_exact_mcmc(y_ma, q_true, n_iter = 60000, burn_in = 10000)
cat(sprintf("  exact MCMC acceptance = %.2f\n", exact$acc_rate))

# ---- history -> ggsmc tidy data ------------------------------------
history_to_tidy <- function(history) {
  do.call(rbind, lapply(seq_along(history), function(t) {
    h <- history[[t]]
    matrix2tidy(h$theta, parameter = "theta", target = t, log_weights = h$log_w)
  }))
}
tidy_abc <- history_to_tidy(fit_abc$history)
tidy_re2 <- history_to_tidy(fit_re2$history)

eps_lab <- function(t) sprintf("iter %d  (eps=%.1f)", t, schedule[t])
target_labeller <- setNames(eps_lab(seq_along(schedule)), seq_along(schedule))

tidy_abc$Method <- "ABC-SMC"
tidy_re2$Method <- "RE-ABC-SMC^2"
tidy_both <- rbind(tidy_abc, tidy_re2)
tidy_both$Method <- factor(tidy_both$Method, levels = c("ABC-SMC", "RE-ABC-SMC^2"))
true_df <- data.frame(Dimension = c(1, 2), tv = theta_true)
cols2  <- c("ABC-SMC" = "#d62728", "RE-ABC-SMC^2" = "#1f77b4")

# exact samples thinned for plotting, replicated across all targets so the
# black reference density appears in every facet
n_targets <- length(schedule)
exact_plot <- exact$samples[seq(1, nrow(exact$samples), length.out = 4000), ]

# =====================================================================
# Static figure
# =====================================================================
pdf("R/theta_evolution.pdf", width = 11, height = 4.2)

## marginal evolution, one page per theta-dimension
for (d in 1:2) {
  sub <- tidy_both[tidy_both$Dimension == d, ]
  exact_rep <- data.frame(
    Target = rep(seq_len(n_targets), each = nrow(exact_plot)),
    Value  = rep(exact_plot[, d], n_targets))
  g <- ggplot(sub, aes(Value, weight = exp(LogWeight),
                       colour = Method, fill = Method)) +
    geom_density(alpha = 0.25) +
    geom_density(data = exact_rep, aes(Value), inherit.aes = FALSE,
                 colour = "black", linewidth = 0.7) +
    geom_vline(data = true_df[true_df$Dimension == d, ],
               aes(xintercept = tv), linetype = 2, colour = "grey40") +
    facet_wrap(~ Target, nrow = 1, scales = "free_y",
               labeller = labeller(Target = target_labeller)) +
    scale_colour_manual(values = cols2) + scale_fill_manual(values = cols2) +
    labs(title = bquote("MA(2): posterior on " * theta[.(d)] *
                        " over iterations  (black = exact posterior; dashed = true value " *
                        .(theta_true[d]) * ")"),
         x = bquote(theta[.(d)]), y = "weighted density") +
    theme_bw(base_size = 10) + theme(legend.position = "top")
  print(g)
}

## joint (theta_1, theta_2): exact posterior contours + final ABC particles
grid <- ma_exact_grid_posterior(y_ma)
gridpost <- transform(grid$grid, dens = grid$weight)
resample_particles <- function(fit, n = 1500) {
  Th <- do.call(rbind, fit$theta)
  Th[sample.int(nrow(Th), n, replace = TRUE, prob = fit$weights), ]
}
pts <- rbind(
  data.frame(theta1 = resample_particles(fit_abc)[, 1],
             theta2 = resample_particles(fit_abc)[, 2], Method = "ABC-SMC"),
  data.frame(theta1 = resample_particles(fit_re2)[, 1],
             theta2 = resample_particles(fit_re2)[, 2], Method = "RE-ABC-SMC^2"))
gj <- ggplot() +
  geom_contour(data = gridpost, aes(theta1, theta2, z = dens),
               colour = "black", bins = 8, linewidth = 0.3) +
  geom_point(data = pts, aes(theta1, theta2, colour = Method),
             alpha = 0.15, size = 0.6) +
  geom_point(aes(x = theta_true[1], y = theta_true[2]), shape = 4,
             size = 3, stroke = 1.2, colour = "grey20") +
  scale_colour_manual(values = cols2) +
  coord_cartesian(xlim = c(-0.5, 1.5), ylim = c(-0.6, 0.9)) +
  guides(colour = guide_legend(override.aes = list(alpha = 1, size = 2))) +
  labs(title = "Joint posterior at the final tolerance (eps = 1) vs exact",
       subtitle = "black contours = exact posterior; x = true theta",
       x = expression(theta[1]), y = expression(theta[2])) +
  theme_bw(base_size = 11) + theme(legend.position = "top")
print(gj)
invisible(dev.off())
cat("Wrote R/theta_evolution.pdf\n")

# =====================================================================
# ggsmc animations over the iterations (Target)
# =====================================================================
anim <- function(tidy, fn, what = c("density", "scatter")) {
  what <- match.arg(what)
  ok <- tryCatch({
    if (what == "density")
      animate_density(tidy, parameter = "theta", dimension = 1,
                      use_initial_points = FALSE, save_filename = fn)
    else
      animate_scatter(tidy, x_parameter = "theta", x_dimension = 1,
                      y_parameter = "theta", y_dimension = 2,
                      use_initial_points = FALSE, alpha = 0.3, save_filename = fn)
    "ok"
  }, error = function(e) conditionMessage(e))
  cat(sprintf("  %-32s -> %s\n", fn, ok))
}
cat("Rendering ggsmc animations (magick) ...\n")
anim(tidy_abc, "R/theta_density_abc_smc.gif",  "density")
anim(tidy_re2, "R/theta_density_re_abc_smc2.gif", "density")
anim(tidy_abc, "R/theta_scatter_abc_smc.gif",  "scatter")
anim(tidy_re2, "R/theta_scatter_re_abc_smc2.gif", "scatter")

# =====================================================================
cat("\nPosterior means (final tolerance eps = 1):\n")
wm <- function(f) { Th <- do.call(rbind, f$theta); colSums(f$weights * Th) }
cat(sprintf("  ABC-SMC       (% .3f, % .3f)\n", wm(fit_abc)[1], wm(fit_abc)[2]))
cat(sprintf("  RE-ABC-SMC^2  (% .3f, % .3f)\n", wm(fit_re2)[1], wm(fit_re2)[2]))
cat(sprintf("  EXACT (MCMC)  (% .3f, % .3f)   <- ground truth\n",
            mean(exact$samples[, 1]), mean(exact$samples[, 2])))
cat(sprintf("  (true theta   (% .3f, % .3f); the exact posterior need not be\n",
            theta_true[1], theta_true[2]))
cat("   centred there at finite n.)\n")
