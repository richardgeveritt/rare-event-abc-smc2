# R implementation of the algorithms in *Rare-event ABC-SMC²*

Model-agnostic R code for the three SMC algorithms in the paper:

| File | Algorithm in paper | Purpose |
|------|--------------------|---------|
| [`rare_event_smc.R`](rare_event_smc.R) | `alg:rare-event-smc` (Rare event SMC) | Estimates the ABC likelihood `l(y\|θ)` for a single `θ` via an inner SMC over the random vector `u`. |
| [`re_abc_smc2.R`](re_abc_smc2.R)       | `alg:abc-smc2` (RE-ABC-SMC²)          | External SMC over `θ` whose likelihoods come from the rare event SMC above. |
| [`abc_smc.R`](abc_smc.R)               | `alg:abc-smc` (ABC-SMC)               | Standard ABC-SMC (Del Moral et al. 2012) — the comparator. |
| [`smc_utils.R`](smc_utils.R)           | —                                     | Shared helpers: weight normalisation, ESS, resampling, CESS, adaptive tolerance (bisection), adaptive number of MCMC moves. |

`example_model_gaussian.R` and `test_run.R` are a throw-away example +
smoke test; replace the model with your own (see below).

### Worked example models

Three example models live alongside the algorithms, each a
`make_*_model()` constructor returning the model `list` described below:

| File | Constructor | Reparameterisation `u` |
|------|-------------|------------------------|
| [`example_model_gaussian.R`](example_model_gaussian.R) | `make_gaussian_model(y)` | half-normal latent normals |
| [`ma.R`](ma.R) | `make_ma_model(y, q)` | the `n+q` i.i.d. MA innovations |
| [`lv.R`](lv.R) | `make_lv_model(y, ...)` | all per-step CLE Brownian increments |

In `ma.R` and `lv.R` the reference `φ(u\|θ) = N(0, I)` is independent of
`θ`, so the move on `u` is a preconditioned Crank–Nicolson (pCN)
proposal (`ru_move`/`du_move`). [`test_models.R`](test_models.R) smoke-tests
both through all three algorithms (run from the project root); `lv.R`
needs the `smfsb` package for the LV data and reaction network.

For the MA model, [`ma_exact_posterior.R`](ma_exact_posterior.R) provides
the **exact (non-ABC) posterior** — the stationary MA(q) Gaussian
likelihood `y ~ N(0, Σ(θ))` with the invertibility prior — via
`ma_loglik_exact()`, an adaptive random-walk MH sampler `ma_exact_mcmc()`,
and a 2-D grid evaluator. This is the ground truth the ABC posteriors
converge to as `ε → 0`. [`visualise_theta_evolution.R`](visualise_theta_evolution.R)
shows the ABC-SMC / RE-ABC-SMC² posteriors evolving over the tolerance
schedule against this exact reference (using the `ggsmc` package), and
[`converge_to_exact.R`](converge_to_exact.R) runs both algorithms
adaptively to `ε = 0.1`: RE-ABC-SMC² with `move = "gaussian"` matches the
exact posterior (mean, sd and correlation), while plain ABC-SMC collapses
to a point at that tolerance on the raw 30-dim data.
[`evidence_model_selection.R`](evidence_model_selection.R) fits MA(2)/MA(3)/MA(4)
to data from each, reporting a 3×3 `log_evidence` table per algorithm
(adaptive to `ε = 1`): both agree on the selected model, but RE-ABC-SMC²'s
evidence is less downward-biased and far more stable than ABC-SMC's in the
hard cases.

```r
source("R/smc_utils.R"); source("R/rare_event_smc.R")
source("R/abc_smc.R");   source("R/re_abc_smc2.R")
```

## What a "model" is

A model is a plain `list` of functions. `θ` and `u` may be any R object
(numeric vector, list, …); a single simulated dataset `x` likewise. The
observed data `y` is captured inside the kernel/`H` closures.

### Needed by **ABC-SMC** (`abc_smc`)

| Field | Signature | Returns |
|-------|-----------|---------|
| `rprior(N)`             | N → list           | `N` draws `θ ~ p` |
| `dprior(θ)`             | θ → scalar         | `log p(θ)` |
| `rproposal(θ)` †        | θ → θ*             | proposed `θ* ~ q(·\|θ)` |
| `dproposal(to, from)` † | (θ,θ) → scalar     | `log q(to\|from)` |
| `simulate(θ, Nx)`       | (θ,Nx) → list      | `Nx` draws `x ~ f(·\|θ)` |
| `log_abc_kernel(x, ε)`  | (x,ε) → scalar     | `log P_ε(y\|x)` |

† Only needed when `adapt_theta_proposal = FALSE`; the default adaptive
proposal (see below) supplies its own symmetric Gaussian random walk.

### Needed by **rare event SMC** and **RE-ABC-SMC²** (reparameterised model)

`x = H(u, θ)`, with all stochasticity in `u ~ φ(·\|θ)`.

| Field | Signature | Returns |
|-------|-----------|---------|
| `rprior`, `dprior`, `rproposal`, `dproposal` | as above | (θ-level) |
| `rphi(θ, Nu)`           | (θ,Nu) → list  | `Nu` draws `u ~ φ(·\|θ)` |
| `dphi(u, θ)`            | (u,θ) → scalar | `log φ(u\|θ)` |
| `H(u, θ)`               | (u,θ) → x      | deterministic transform |
| `log_abc_kernel(x, ε)`  | (x,ε) → scalar | `log P_ε(y\|x)` |
| `ru_move(u, θ)`         | (u,θ) → u*     | MCMC proposal on `u` (kernel `K_t`) — needed for `move="mh"` |
| `du_move(to, from, θ)`  | (u,u,θ) → scalar | `log` proposal density; return `0` if symmetric |
| `grad_loglik_u(u, θ, ε)` | (u,θ,ε) → vector | `∇_u log P_ε(y\|H(u,θ))` — needed for `move="pcn-mala"` |
| `rtarget_gaussian(θ, ε, n)` | (θ,ε,n) → list | `n` exact iid draws from `π(u)` — needed for `move="gaussian"` |

The `u`-move (kernel `K_t`, invariant `P_ε(y\|H(u,θ)) φ(u\|θ)`) is selected
by the `move` argument (see "Choosing the u-move" below). The default
`"mh"` is a Metropolis–Hastings step built from `ru_move`/`du_move`
(Prangle 2016 uses a slice sampler; any invariant kernel is fine). For the
"split randomness" variant of Section "Splitting different sources of
randomness", let `ru_move` move only the part `u_s` and leave `u_r` fixed.

`grad_loglik_u`/`rtarget_gaussian` are only required for the corresponding
non-default moves. `ma.R` provides both (its `H` is linear, so the
`u`-target is exactly Gaussian); `lv.R` provides neither, so LV uses
`move="mh"`.

## Running

Both samplers accept either an explicit decreasing tolerance schedule or
adaptive tolerance selection (CESS bisection to `beta * N`, the default):

```r
# adaptive tolerances down to epsilon_final
fit <- re_abc_smc2(model, N_theta = 1000, Nu = 100,
                   epsilon_final = 1.5, beta = 0.9)

# or an explicit schedule
fit <- abc_smc(model, N_theta = 1000, Nx = 50,
               epsilon_schedule = c(20, 12, 7, 4, 2.5, 1.5))
```

Common arguments: `alpha` (resample when `ESS < alpha·N`), `beta` (target
CESS fraction), `adapt_nmoves`/`c_move`/`max_moves` (adaptive number of
MCMC sweeps, South et al. 2019, `c = 0.2`), `resample_scheme`
(`"multinomial"` as in the paper, or `"systematic"`).

### Choosing the u-move (rare event SMC / RE-ABC-SMC²)

`re_abc_smc2` (and `re_smc_run`/`re_smc_step`) take a `move` argument
selecting the inner move on `u`:

| `move` | description | model needs |
|--------|-------------|-------------|
| `"pcn-mala"` (default) | gradient-based pCN-MALA (∞-MALA); step size adapted to ≈0.574 acceptance and carried across SMC steps (`move_step0` sets the start) | `grad_loglik_u` |
| `"mh"` | generic Metropolis–Hastings (e.g. the pCN move in `ma.R`/`lv.R`) | `ru_move`, `du_move` |
| `"gaussian"` | replace every `u`-particle with an **exact independent draw** from the Gaussian `u`-target — no MCMC (only valid when `H` is linear and the kernel Gaussian) | `rtarget_gaussian` |

The default is `"pcn-mala"`; if the model does not supply the piece a move
needs (`grad_loglik_u` for pcn-mala, `rtarget_gaussian` for gaussian) it
**falls back silently to `"mh"`**, so gradient-free models (`lv.R`,
`example_model_gaussian.R`) still run with the default.

```r
fit <- re_abc_smc2(model_ma, N_theta = 200, Nu = 60,
                   epsilon_schedule = c(20, 11, 6, 3, 1))   # pcn-mala by default
```

On the MA model the gradient/exact moves markedly reduce the variance of
the inner likelihood estimate (and hence sharpen the outer posterior)
relative to `"mh"`. The standalone comparison of RWM / pCN / MALA /
pCN-MALA / HMC for this `u`-target is in
[`mcmc_u_comparison.R`](mcmc_u_comparison.R) (writes
`mcmc_u_comparison.pdf`).

### Adaptive theta proposal

By default (`adapt_theta_proposal = TRUE`) the Metropolis–Hastings move on
`theta` uses a **multivariate** Gaussian random walk whose covariance is
the **weighted sample covariance of the theta-population, measured just
before the resampling step** (the unbiased weighted-sample estimator, as in
`stats::cov.wt`; full covariance matrix, so cross-component correlations are
respected via a Cholesky factor). The scale is held fixed for the whole
move step, so the proposal is symmetric and `model$rproposal` /
`model$dproposal` are **not used** in this mode. The covariance is scaled by
the Roberts–Gelman–Gilks optimal factor: the proposal standard deviations
are `2.38/sqrt(d)` times the weighted sample sd (`d` = dim of `theta`), i.e.
the proposal covariance is `(2.38^2/d)·Σ̂`. This is the default
(`proposal_scale = NULL`); pass a numeric `proposal_scale` to use that fixed
value as the scale on the sd instead.

Set `adapt_theta_proposal = FALSE` to fall back to the model-supplied
`rproposal`/`dproposal` instead.

Return value (named list): `theta` (list of particles), `weights`,
`log_evidence` (model-evidence estimate), `epsilon`/`epsilon_schedule`,
`ess`, `acc_rate`, `n_iterations`; `re_abc_smc2` also returns `states`
(the inner rare-event SMCs). Posterior expectation of a scalar `θ`:
`sum(fit$weights * unlist(fit$theta))`.

`re_smc_run(model, theta, Nu, epsilon_schedule)` runs the rare event SMC
alone; `$log_lik` is `log l̄(y\|θ) = log ∏_t Σ_n w̃ⁿ_t`.

## Validation (smoke test)

`Rscript R/test_run.R` runs all three on the Gaussian example. Two checks
the code is wired up correctly:

* ABC-SMC and RE-ABC-SMC² — which use **different** likelihood estimators
  — agree on the posterior mean and on `log_evidence`.
* The rare event SMC `log_lik` matches a brute-force importance-sampling
  estimate of the same ABC likelihood across a grid of `θ`.
