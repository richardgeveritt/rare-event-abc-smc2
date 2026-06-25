# =====================================================================
# evidence_experiments.R
#
# Replicated MA log-evidence experiment + dimension sweep, comparing
# ABC-SMC with RE-ABC-SMC^2 (move = "gaussian").
#
#   * full 3x3 model x model log-evidence table at n = 100, with error
#     bars from several replicates;
#   * a dimension sweep (correct-model fits) showing how the gap between
#     ABC-SMC and RE-ABC-SMC^2 log-evidence grows with the data length n.
#
# PERSISTENCE / ROBUSTNESS
#   Every individual fit is checkpointed to its own .rds file in
#   R/evidence_results/ as soon as it finishes.  The script is therefore
#   * resumable    : re-running skips fits whose .rds already exists;
#   * sleep-safe   : if the laptop sleeps the process is merely paused;
#                    completed fits are already on disk regardless.
#   It runs the pending fits in PARALLEL (parallel::mclapply, fork), then
#   aggregates everything on disk into tables + a plot.  Re-run as often
#   as you like; it continues where it left off and re-aggregates.
#
# Recommended (detached, survives closing the terminal/IDE):
#   nohup Rscript R/evidence_experiments.R > R/evidence_experiments.log 2>&1 &
# =====================================================================

source("R/smc_utils.R")
source("R/rare_event_smc.R")
source("R/abc_smc.R")
source("R/re_abc_smc2.R")
source("R/ma.R")
suppressMessages({ library(parallel) })

# ------------------------------------------------------------------ config
# (env vars allow a cheap smoke test without editing the defaults)
orders     <- c(2, 3, 4)
reps       <- as.integer(Sys.getenv("EVID_REPS",  "5"))
N_theta    <- as.integer(Sys.getenv("EVID_NTHETA", "500"))
eps_fin    <- 1
RE_Nu      <- 40
ABC_Nx     <- 25
full_n     <- as.integer(Sys.getenv("EVID_FULL_N", "100"))   # full 3x3 table here
sweep_n    <- as.integer(strsplit(Sys.getenv("EVID_SWEEP_N", "30,100,200,300"),
                                  ",")[[1]])                  # dims for the sweep
results_dir <- Sys.getenv("EVID_DIR", "R/evidence_results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

# fixed true coefficients (reproducible; non-negligible top coefficient)
set.seed(123)
true_theta <- lapply(orders, function(q) {
  repeat { th <- as.numeric(simulate_prior_maq(1, q)); if (abs(th[q]) > 0.3) return(th) }
})
names(true_theta) <- as.character(orders)

# deterministic data set for each (n, data_q)
make_data <- function(n, data_q) {
  set.seed(10000 * data_q + n)
  make_ma_model(rep(0, n), data_q)$H(rnorm(n + data_q), true_theta[[as.character(data_q)]])
}

# (n, data_q, fit_q) configurations: full table at full_n, diagonal elsewhere
configs <- list()
for (dq in orders) for (fq in orders)
  configs[[length(configs) + 1]] <- c(n = full_n, data_q = dq, fit_q = fq)
for (nn in setdiff(sweep_n, full_n)) for (q in orders)
  configs[[length(configs) + 1]] <- c(n = nn, data_q = q, fit_q = q)
configs <- unique(configs)

# full job list: (config) x algorithm x replicate
# (default both RE moves; override with EVID_ALGOS="ABC-SMC,RE-pcnmala" etc.)
algos <- strsplit(Sys.getenv("EVID_ALGOS", "ABC-SMC,RE-gaussian,RE-pcnmala"),
                  ",")[[1]]
algo_tag <- c("ABC-SMC" = "abc", "RE-gaussian" = "reg", "RE-pcnmala" = "rpm")
jobs  <- list()
for (cf in configs) for (al in algos) for (r in seq_len(reps)) {
  id <- sprintf("n%03d_d%d_f%d_%s_r%02d", cf["n"], cf["data_q"], cf["fit_q"],
                algo_tag[al], r)
  jobs[[id]] <- list(id = id, n = unname(cf["n"]), data_q = unname(cf["data_q"]),
                     fit_q = unname(cf["fit_q"]), algo = al, rep = r,
                     file = file.path(results_dir, paste0(id, ".rds")))
}
# run cheap (small-n) fits first so low-n results land early
jobs <- jobs[order(vapply(jobs, function(j) j$n, numeric(1)))]

# ------------------------------------------------------------------ one fit
run_job <- function(job) {
  if (file.exists(job$file)) return(invisible(NULL))        # resume: already done
  y     <- make_data(job$n, job$data_q)
  model <- make_ma_model(y, job$fit_q)
  set.seed(1e6 * job$rep + 1000 * job$data_q + 100 * job$fit_q + job$n)
  t0  <- Sys.time()
  fit <- tryCatch({
    if (job$algo == "ABC-SMC")
      abc_smc(model, N_theta = N_theta, Nx = ABC_Nx, epsilon_final = eps_fin,
              beta = 0.9, max_steps = 150L, max_moves = 10L)
    else if (job$algo == "RE-gaussian")
      re_abc_smc2(model, N_theta = N_theta, Nu = RE_Nu, epsilon_final = eps_fin,
                  beta = 0.9, move = "gaussian", max_steps = 150L)
    else                                              # RE-pcnmala
      re_abc_smc2(model, N_theta = N_theta, Nu = RE_Nu, epsilon_final = eps_fin,
                  beta = 0.9, move = "pcn-mala", max_steps = 150L, max_moves = 20L)
  }, error = function(e) e)
  secs <- as.numeric(Sys.time() - t0, units = "secs")
  res <- if (inherits(fit, "error"))
    data.frame(id = job$id, n = job$n, data_q = job$data_q, fit_q = job$fit_q,
               algo = job$algo, rep = job$rep, logZ = NA_real_, eps = NA_real_,
               iters = NA_integer_, secs = secs, status = conditionMessage(fit))
  else
    data.frame(id = job$id, n = job$n, data_q = job$data_q, fit_q = job$fit_q,
               algo = job$algo, rep = job$rep, logZ = fit$log_evidence,
               eps = tail(fit$epsilon, 1), iters = fit$n_iterations,
               secs = secs, status = "ok")
  tmp <- paste0(job$file, ".tmp")
  saveRDS(res, tmp); file.rename(tmp, job$file)             # atomic checkpoint
  invisible(NULL)
}

# ------------------------------------------------------------------ run
# EVID_AGGREGATE_ONLY=1 snapshots the current checkpoints (no fitting);
# safe to run while the main job is still going (reads only).
aggregate_only <- nzchar(Sys.getenv("EVID_AGGREGATE_ONLY", ""))
pending <- Filter(function(j) !file.exists(j$file), jobs)
cat(sprintf("%d jobs total | %d done | %d pending%s\n",
            length(jobs), length(jobs) - length(pending), length(pending),
            if (aggregate_only) "  (aggregate-only: not running them)" else ""))
if (!aggregate_only && length(pending) > 0) {
  ncores <- as.integer(Sys.getenv("EVID_CORES",
                                  as.character(max(1L, parallel::detectCores() - 1L))))
  cat(sprintf("Running %d pending fits on %d cores ...\n", length(pending), ncores))
  RNGkind("L'Ecuyer-CMRG")
  invisible(parallel::mclapply(pending, run_job,
                               mc.cores = ncores, mc.preschedule = FALSE))
}

# ------------------------------------------------------------------ aggregate
files <- list.files(results_dir, pattern = "\\.rds$", full.names = TRUE)
res   <- do.call(rbind, lapply(files, readRDS))
write.csv(res, "R/evidence_results_raw.csv", row.names = FALSE)
ok    <- res[res$status == "ok" & is.finite(res$logZ), ]
cat(sprintf("\n%d/%d fits succeeded (raw rows in R/evidence_results_raw.csv).\n",
            nrow(ok), nrow(res)))

summ <- aggregate(logZ ~ n + data_q + fit_q + algo, ok,
                  function(z) c(mean = mean(z), sd = sd(z), nrep = length(z)))
summ <- do.call(data.frame, summ)
names(summ) <- c("n", "data_q", "fit_q", "algo", "mean", "sd", "nrep")
write.csv(summ, "R/evidence_summary.csv", row.names = FALSE)

lab <- paste0("MA(", orders, ")")
cell <- function(n, dq, fq, al) {
  r <- summ[summ$n == n & summ$data_q == dq & summ$fit_q == fq & summ$algo == al, ]
  if (nrow(r) == 0) "    NA    " else sprintf("%7.1f(%.1f)", r$mean, r$sd)
}
for (al in algos) {
  cat(sprintf("\n== %s : mean log-evidence (sd over reps), n=%d ==\n", al, full_n))
  cat(sprintf("%-8s %s\n", "data\\fit", paste(sprintf("%-12s", lab), collapse = "")))
  for (dq in orders)
    cat(sprintf("%-8s %s\n", lab[match(dq, orders)],
        paste(sprintf("%-12s", vapply(orders, function(fq) cell(full_n, dq, fq, al), "")),
              collapse = "")))
}

# ------------------------------------------------------------------ sweep plot
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  dia <- summ[summ$data_q == summ$fit_q, ]
  dia$Model <- factor(paste0("MA(", dia$data_q, ") data"))
  p1 <- ggplot(dia, aes(n, mean, colour = algo)) +
    geom_line() + geom_point() +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 4) +
    facet_wrap(~ Model, scales = "free_y") +
    labs(title = "Correct-model log-evidence vs data length n (mean +/- sd over reps)",
         x = "n (data length)", y = "log-evidence", colour = "") +
    theme_bw(base_size = 11) + theme(legend.position = "top")

  # bias of each RE variant relative to ABC-SMC: (RE_variant - ABC) vs n
  re_variants <- intersect(c("RE-gaussian", "RE-pcnmala"), unique(dia$algo))
  gap <- do.call(rbind, lapply(split(dia, list(dia$n, dia$data_q), drop = TRUE),
    function(d) {
      a <- d$mean[d$algo == "ABC-SMC"]; sa <- d$sd[d$algo == "ABC-SMC"]
      if (!length(a)) return(NULL)
      do.call(rbind, lapply(re_variants, function(v) {
        r <- d$mean[d$algo == v]; sr <- d$sd[d$algo == v]
        if (length(r))
          data.frame(n = d$n[1], data_q = d$data_q[1], variant = v, gap = r - a,
                     gsd = sqrt(sum(c(sa, sr)^2, na.rm = TRUE)))
      }))
    }))
  gap$Model <- factor(paste0("MA(", gap$data_q, ") data"))
  p2 <- ggplot(gap, aes(n, gap, colour = Model, linetype = variant)) +
    geom_line() + geom_point() +
    labs(title = "ABC-SMC evidence bias vs n  (RE minus ABC-SMC log-evidence)",
         subtitle = "the RE moves are the low-variance reference; the gap ~ ABC-SMC downward bias",
         x = "n (data length)", y = "log-evidence gap (nats)",
         colour = "", linetype = "RE move") +
    theme_bw(base_size = 11) + theme(legend.position = "top")

  # cost of MCMC vs exact draw: (RE-gaussian - RE-pcnmala) vs n
  plots <- list(p1, p2)
  if (all(c("RE-gaussian", "RE-pcnmala") %in% dia$algo)) {
    pen <- do.call(rbind, lapply(split(dia, list(dia$n, dia$data_q), drop = TRUE),
      function(d) {
        g <- d$mean[d$algo == "RE-gaussian"]; m <- d$mean[d$algo == "RE-pcnmala"]
        if (length(g) && length(m))
          data.frame(n = d$n[1], Model = paste0("MA(", d$data_q[1], ") data"),
                     diff = g - m)
      }))
    plots[[3]] <- ggplot(pen, aes(n, diff, colour = Model)) +
      geom_line() + geom_point() + geom_hline(yintercept = 0, colour = "grey70") +
      labs(title = "Cost of MALA-pCN vs the exact Gaussian move (RE-gaussian - RE-pcnmala)",
           subtitle = "near zero => pcn-mala matches the exact-draw evidence",
           x = "n (data length)", y = "log-evidence difference (nats)", colour = "") +
      theme_bw(base_size = 11) + theme(legend.position = "top")
  }

  pdf("R/evidence_sweep.pdf", width = 9, height = 4.5)
  for (p in plots) print(p)
  invisible(dev.off())
  cat("\nWrote R/evidence_sweep.pdf, R/evidence_summary.csv, R/evidence_results_raw.csv\n")
}
