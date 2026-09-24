## Architecture developer reference

Code and data for the Master's thesis of **Juan Manuel Oliver** (MSc Artificial Intelligence, Big Data Analytics, KU Leuven, 2025–26). The project builds a calibrated NBA shot-make-probability model from 2015–16 SportVU optical tracking + play-by-play data, trains it both centrally and under a cross-silo **federated learning** protocol (30 teams as clients), and derives downstream applications: **Points Over Expectation (POE)** for shooters, defenders, zones, and 5-man lineups.

## Repository layout

Top level:

| Path | What it is |
|---|---|
| `allgames.txt` | 636 SportVU game archive names (2015–16 regular season), consumed by `process_batch.py`. |
| `requirements.txt` | Python dependencies (xgboost, flwr, scikit-learn, py7zr, …). |
| `src/` | Python pipeline (feature extraction → model → POE → federated). |
| `clusters/` | R project for player archetype clustering. |
| `data/` | Datasets, saved models |
| `results/` | Metric tables, POE leaderboards, tuning history, and `results/federated/`. |
| `figures/` | All generated figures (result figures + thesis figure outputs, consolidated here). |

### `clusters/` — player archetype clustering (R / RStudio)

Standalone R project (`renv` lockfile included) that produces the soft archetype features used by the Python pipeline.

| File | Purpose |
|---|---|
| `scrape.Rmd` | Scrapes NBA.com hidden stats API for player-tracking defense tables → `data/defensive_tracking.csv`. |
| `shooters.Rmd` | Offensive archetypes: PCA + K-means + **GMM (BIC-selected K=4)** on shot-creation profile (USG, 3PAr, %Ast, FTr, FG% by zone). Outputs `shooter_archetypes_15_16.csv`, `gmm_soft_labels_15_16.csv` (Primary_Creator, Spacer, Mid-Interior, Rim_Center). |
| `defense.Rmd` | Defensive archetypes, same method. Outputs `defender_archetypes_15_16.csv`, `gmm_soft_labels_def_15_16.csv` (Paint_Anchors, Perimeter Guards, Def_liability, Switch_Wing). |
| `data/` | Basketball-Reference exports: `ad.csv`/`pos.csv`/`shot.csv` , `ad_14.csv`/`pos_14.csv`/`shot_14.csv`, `defensive_tracking.csv`, `opp_shooting_by_zone.csv`. |
| `pca_loadings.csv`, `gmm_*_profiles_*.csv` | Cluster interpretation artifacts. |

### `src/` — main Python pipeline

| File | Purpose |
|---|---|
| `game.py` | `Game` class: downloads a game's SportVU 7z from the `sealneaward/nba-movement-data` mirror + its PBP CSV, parses moments into a DataFrame, aligns clocks, exposes helpers (frames, commentary, formation detection). |
| `kinematics.py` | Savitzky-Golay velocity/acceleration estimation per player; `time_to_reach` closeout-time model (max speed 22 ft/s, max accel 10 ft/s²). |
| `spatial.py` | Team convex-hull spacing, Voronoi space control, delta-distance/delta-time control maps. |
| `shot_features.py` | Core extractor. For each PBP shot event: recovers the release frame (ball-z apex ≥ 9 ft, walk-back to z ≤ 10 ft & ball–shooter dist ≤ 2.5 ft), then computes ~37 model features: geometry (dist, x, y, angle), defender pressure (closest/second defender distance-angle-time, tight-contest counts), kinematic decomposition (parallel/perpendicular velocity & acceleration for shooter and defender), release mechanics (height, speed, angle, x, y), tempo (shot clock, touch time, catch-and-shoot), spacing (hull ratio), and the 8 GMM archetype probabilities. Dunks/tips flagged via PBP regex. |
| `process_batch.py` | Multiprocessing batch driver over `allgames.txt` → `data/shot_features_full.csv` (rename/copy to `_valid2` after archetype columns are applied). |
| `train_xgboost.py` | Canonical centralized model on `data/shot_features_valid2_type.csv` → `data/xgb_shot_model.json`. Defines `DATA_PATH` and `METADATA_COLS` (same features as the federated model). Held-out test = the federated global test games (`GroupShuffleSplit` on `game_id`, 15%, seed 42); 15% of the remaining games for early stopping on log-loss; evaluation focused on calibration. |
| `tune_xgboost.py` | Random search (30 candidates × GroupKFold(5)) scored by CV log-loss on the same train/test games as `train_xgboost.py`; refits best config → `data/xgb_shot_model_tuned.json`, `results/best_params.json`. |
| `compute_poe.py` | Out-of-fold (GroupKFold over games, 1150 trees ≈ the canonical model's early-stopping point) P(make) for every shot → per-shot POE (`shot_value × (made − xMake)`, shot value from PBP "3PT" regex) → player leaderboard (≥100 shots). |
| `compute_def_poe.py` | Defender POE (points suppressed below expectation for the closest defender) + offensive-vs-defensive archetype matchup heatmap. |
| `correlate_poe.py` | Correlates POE with Basketball-Reference advanced metrics (TS%, PER, OBPM, …) incl. partial correlations controlling for USG%. Reads `results/poe_leaderboard.csv` and `clusters/data/{ad,pos}.csv`; writes correlation tables to `results/`. |
| `thesis_results.py` | One-shot generator of Results-chapter tables and figures (headline metrics vs baselines, per-zone metrics, calibration, feature importance, POE leaderboard, case-study shot charts). Reads the canonical dataset (`train_xgboost.DATA_PATH`). |
| `plot_per_zone_poe.py` | Per-zone POE decomposition figure (Results section). Reads `results/poe_per_shot.csv`. |
| `visualization.py` | Court drawing, frame rendering, game animation utilities (used for Figure 3.1-style renders). |

### `src/federated/` — Flower federated XGBoost

| File | Purpose |
|---|---|
| `nba_federated/task.py` | Data loading/partitioning from `data/shot_features_valid2_type.csv`. Single cached game-disjoint global 85/15 train/test split (seed 42, shared by all clients & server, identical to `train_xgboost.py`); partitions: `team` (30 non-IID clients = franchises) or `iid` (random control); each client holds out 20% of its games for local validation. |
| `nba_federated/client_app.py` | Flower client: trains 1 tree/round on local data from the global booster. |
| `nba_federated/server_app.py` | Flower server: `FedXgbBagging` (primary) or `FedXgbCyclic` (ablation; deterministic round-robin + central-eval adapter). Logs per-round test metrics for curves only and saves the final booster → `results/federated/`. No checkpoint is chosen on test data. |
| `nba_federated/hardening.py` | Layer-1 defences, off by default (`harden-strip`, `public-bins` in `pyproject.toml`): zero the per-node stats prediction never reads (`sum_hessian`, `loss_changes`, internal weights), and train on a fixed public bin grid (ranges from physical limits) with thresholds snapped to it. Predictions on raw features are unchanged; public bins cost no measurable accuracy. |
| `leakage_attack.py` | Reconstruction attack on round-1 trees: an honest-but-curious server recovers each team's exact shot and make counts per tree region (also through the leaf-only DP prototype), and — after layer-1 hardening — still each region's make rate from leaf values alone. → `leakage_attack_summary.csv`, `leakage_attack_leaves.csv`. |
| `nba_federated/hist_gbdt.py` | **Histogram protocol** engine: all teams grow one shared tree per round from summed per-bin gradient/hessian histograms on the public grid (exactly centralized XGBoost `hist` without DP, verified to 1e-7), optional distributed Gaussian DP on the histograms (`split_mode` `hist` or data-independent `random` structure), export to a standard XGBoost booster. |
| `nba_federated/hist_client.py`, `hist_server.py` | Flower ClientApp / ServerApp for the histogram protocol: one round per tree level, federated validation after each tree, optional SecAgg+ (`hist-secagg`) so the server only sees the sum over teams, and desync detection (clients report their applied tree count; the server resends the model on a mismatch). |
| `sim_histogram.py` | In-process runs of the histogram protocol on the 30 team silos: `--utility` (no DP) and `--dp-sweep` (privacy/utility frontier; parts merged with `--merge`) → `hist_utility.csv`, `hist_dp_frontier.csv`. |
| `leakage_attack_hist.py` | The same reconstruction against the histogram protocol's first-tree releases: exact per-team counts in every (feature, bin) cell without SecAgg; only league figures with SecAgg+; noisy under DP; plus the formal per-team ε if SecAgg is dropped. → `leakage_attack_hist_summary.csv`. |
| `per_team_benefit.py` | Each team scored on its own held-out shots under four models (alone / histogram protocol / bagging / centralized), with a game-level bootstrap of the federated − alone gap; teams with fewer than `--min-test-games` (default 3) held-out games are excluded from the reported stats (`included` column) → `per_team_benefit.csv`, `per_team_benefit.png`. |
| `team_heterogeneity.py` | How non-IID the team silos are, against the IID control: make-rate spread (permutation test), per-feature Jensen-Shannon divergence on the public bins, and a train-on-one-team / score-every-team transfer matrix → `team_heterogeneity*.csv`, `team_heterogeneity.png`. |
| `dp_audit.py` | Membership-inference audit of the released model (members = clients' train shots, non-members = their validation shots) with a Clopper-Pearson empirical-ε lower bound → `dp_audit.csv`. |
| `plot_leakage.py` | Reconstruction figure: each team-region's true FG% against what the server recovers, as published / after stripping / under DP → `results/federated/leakage_attack.png`. |
| `plot_privacy_utility.py` | Privacy/utility figure (AUC and Brier vs ε, shot- vs player-level DP, reference lines) → `results/federated/privacy_utility.png` (copied to `assets/`). |
| `nba_federated/dp.py` | RDP accountant for the Gaussian mechanism (used by `hist_gbdt.py`), plus the superseded leaf-perturbation prototype for the bagging protocol, which is NOT a valid guarantee (tree structure and node statistics are sent in the clear; see `leakage_attack.py`). |
| `pyproject.toml` | Flower app config + XGBoost hyperparameters (depth 5, eta 0.05, subsample/colsample 0.8, min_child_weight 10, λ 2.0), shared with the matched centralized baseline. |
| `run_seeds.py` | Runs the 4 configs (bagging/cyclic × iid/team) × 5 seeds {42, 7, 123, 2024, 99}, renaming outputs per run. |
| `evaluate_federated.py` | Produces every federated number: selects each run's round on federated validation log-loss, trains the matched centralized baseline (same params, union of client train splits) and the local-only baselines, game-level cluster bootstrap CIs → `eval_summary.csv/.tex`, `eval_per_run.csv`, `eval_curves.csv`, `federated_convergence.png`, `*_model_selected.json`. |

### `data/` — datasets and models

| File | Purpose |
|---|---|
| `tracking/` | Three sample raw SportVU game JSONs. |
| `shot_features_valid2_type.csv` | **Canonical training file** — 97,997 shots / 631 games × 65 cols: `valid2` plus 18 PBP shot-type flags (`stype_*`, replacing `is_dunk_or_tip`). Built by `src/experiments/add_shot_type.py`. Used by every model: `train_xgboost.py`, `tune_xgboost.py`, `compute_poe.py`, the federated pipeline. |
| `shot_features_valid2.csv` | Base feature file (97,997 shots × 48 cols, named 4+4 GMM archetype soft labels); input to the `src/experiments/add_*.py` variant builders. |
| `shot_features_valid.csv` | Earlier extraction pass: 97,826 shots / 630 games (the counts quoted in the thesis data section). |
| `xgb_shot_model.json` | Canonical centralized booster (`train_xgboost.py`). |
| `xgb_shot_model_tuned.json` | Tuned centralized booster (`tune_xgboost.py`). |
| `lineup_poe.csv`, `lineup_poe_offense.csv`, `lineup_poe_defense.csv` | 5-man lineup POE tables (built in `spa.ipynb`). |
| `players.csv`, `pbp/` | Support lookups / sample PBP + `EVENTMSGTYPE` code reference (`pbpevents.txt`). |

### `results/` — experiment outputs

Canonical POE outputs (`poe_per_shot.csv`, `poe_leaderboard.csv`), headline/per-zone metric tables, tuning history + best params, correlations with advanced metrics, and `federated/` with per-run test curves, final and validation-selected boosters, and the `eval_*` tables from `evaluate_federated.py`. Pre-2026-09-19 outputs (test-selected checkpoints) are archived in `federated_archive_pre_20260919/` and `legacy_valid2_pre20260919/`.

### `figures/` 

`figures/` is the single output directory for every generated figure (calibration, convergence, heatmaps, leaderboards, plus all thesis `build_*` outputs — PNG + PDF). 

## Reproducing the pipeline

```bash
pip install -r requirements.txt

# 1. (Optional) rebuild archetypes: open clusters/clusters.Rproj, run scrape.Rmd → shooters.Rmd → defense.Rmd

# 2. Extract shot features (downloads games; long-running)
python src/process_batch.py --start 0 --end 636 --output data/shot_features_full.csv

# 3. Centralized model
python src/train_xgboost.py
python src/tune_xgboost.py --n-iter 30

# 4. POE applications (write to results/)
PYTHONPATH=src python src/poe/compute_poe.py
PYTHONPATH=src python src/poe/compute_def_poe.py
PYTHONPATH=src python src/poe/plot_per_zone_poe.py
PYTHONPATH=src python src/poe/correlate_poe.py

# 5. Federated experiments (needs flwr; ~hours)
cd src/federated
pip install -e .
python run_seeds.py                 # bagging/cyclic: 4 configs × 5 seeds
# Flower runs the app from a bundled copy in ~/.flwr/apps, where data/ is not reachable:
# always pass data-path (or set $NBA_FEDERATED_DATA).
D=<repo>/data/shot_features_valid2_type.csv
flwr run . --run-config "protocol='histogram' seed=42 data-path='$D'"                   # histogram protocol
flwr run . --run-config "protocol='histogram' hist-secagg=true seed=42 data-path='$D'"  # … with SecAgg+
flwr run . --run-config "protocol='histogram' hist-secagg=true hist-split-mode='random' dp-epsilon=1.0 \
    seed=42 data-path='$D'"                                                             # … + distributed DP
cd ../..
python src/federated/sim_histogram.py --utility --seeds 42 7 123 2024 99   # histogram protocol, 5 seeds
python src/federated/sim_histogram.py --fixed --seeds 42 7 123 --tag all  # DP headline: shot + player level
python src/federated/sim_histogram.py --dp-sweep --tag all                 # tuned DP sweep (hours; see --help)
python src/federated/sim_histogram.py --merge
python src/federated/evaluate_federated.py   # every federated number, table and figure
python src/federated/leakage_attack.py       # reconstruction attack on bagging trees
python src/federated/leakage_attack_hist.py  # … and on the histogram protocol (± SecAgg+, ± DP)
python src/federated/plot_leakage.py         # reconstruction figure (README / abstract)
python src/federated/plot_privacy_utility.py # privacy/utility figure
python src/federated/per_team_benefit.py     # per-team incentive analysis
python src/federated/team_heterogeneity.py   # how non-IID the silos are
python src/federated/dp_audit.py --merge     # membership-inference audit (run per setting, then merge)
# DP variants, all via --override on the fixed config:
#   subsample=0.2 dp_amplify=true   (subsampling amplification; needs SecAgg+)
#   dp_tolerated_collusion=5        (guarantee survives 5 colluding/dropped teams)
#   dp_discrete=true                (discrete Gaussian on the SecAgg+ lattice)

# 6. Thesis tables/figures (retrains the canonical model + POE)
PYTHONPATH=src python src/thesis_results.py
```

## Headline results (2015–16, game-disjoint test set)

| Model | Brier | Log loss | ROC-AUC |
|---|---|---|---|
| Constant (base rate) | 0.2472 | 0.6875 | 0.500 |
| Distance-only logistic | 0.2411 | 0.6753 | 0.595 |
| XGBoost (full features, canonical) | 0.2042 | 0.5914 | 0.732 |
| Matched centralized (federated params, union of client train splits) | 0.2050 | 0.5932 | 0.729 |
| Federated — histogram protocol, team silos (5 seeds) | 0.2047 | — | 0.730 |
| Federated — tree bagging / cyclic, 4 configs (5 seeds) | 0.2181–0.2206 | 0.624–0.631 | 0.674–0.685 |
| Histogram protocol + distributed DP, shot-level, ε = 1 / 4 (δ = 1e-5, fixed config) | 0.2183 / 0.2133 | — | 0.688 / 0.704 |
| Histogram protocol + distributed DP, player-level, ε = 8 / 16 | 0.2223 / 0.2182 | — | 0.677 / 0.690 |
| One team alone (mean of 30) | 0.2249 | 0.6403 | 0.659 |

Hyperparameter tuning barely moves the centralized model: a 30-candidate random search
(`tune_xgboost.py`, GroupKFold(5) on the same training games) reaches Brier 0.2039 / log
loss 0.5902 / AUC 0.7325 against 0.2043 / 0.5914 / 0.7315 for the hand-set configuration —
−0.2% relative Brier (95% game-level bootstrap CI [−0.4%, −0.0%]), AUC indistinguishable.
The reported model stays `train_xgboost.py`'s, whose hyperparameters the federated
protocols share; `results/best_params.json` records the search.

Every team gains from joining: on its own held-out shots the histogram protocol beats a
model trained on that team alone for every team (mean +0.062 AUC, worst +0.034; bagging +0.016),
excluding three teams that appear in fewer than three held-out games; including them the mean is
+0.063 and all 30 still gain. The silos are heterogeneous in composition but not in the task — make rates span
8.5 points (1.8× a random split, permutation p = 0.0005) and archetype features diverge
~200× the IID control, while a team's own model is only +0.009 AUC better on its own shots
than the other 29 teams' models, which is why the team partition scores like the IID control.

The histogram protocol matches centralized training (relative Brier −0.2%, 95% game-level bootstrap CI [−0.4%, +0.1%]); tree bagging costs +6.4% [+5.7%, +7.0%] (team silos) to +7.7% (cyclic, IID). Team silos are not measurably worse than the IID control. The histogram protocol has been run end to end in Flower for all five seeds, not only in
the in-process simulation (~2,000-3,300 message rounds each, one per tree level):
Brier 0.2047 ± 0.0001 / AUC 0.7307 ± 0.0009, against 0.2047 ± 0.0002 / 0.7305 ± 0.0003 for
the simulation, and −0.17% relative Brier [−0.39%, +0.05%] against the matched centralized
models — the same headline either way (`hist_flower_vs_sim.csv`). Per-seed differences come
from the client-side subsampling RNG, seeded per (team, tree, level) in the Flower client.
Short runs match the engine exactly (max |Δp| 1e-16), and with SecAgg+ to 1e-7.

Two operational notes from those runs. A Flower run executes from a bundled copy of the app
in `~/.flwr/apps`, where the repo's `data/` is not reachable: pass `data-path` in the run
config (or set `$NBA_FEDERATED_DATA`). And if a client's Ray actor restarts — e.g. under
memory pressure — it loses `context.state` and would keep fitting stale residuals, which
wrecks calibration while AUC still creeps up; clients therefore report how many trees they
have applied, the server resends the full model on a mismatch, and the count of such events
is written to the curve CSV (`desyncs`).

Privacy extras, all measured with the same fixed config (`hist_dp_fixed_summary.csv`,
columns `subsample`/`dp_amplify`/`dp_tolerated_collusion`/`dp_discrete`): crediting the
Poisson subsampling the clients already do (sampled-Gaussian accountant, sound only under
SecAgg+) gives ε = 1 shot-level AUC 0.693 instead of 0.688, and does not help at player
level, where whole players must be sampled (0.652 vs 0.656 at ε = 4); tolerating 5 (10)
colluding or dropped teams costs 0.002 (0.004) AUC at ε = 1; discrete-Gaussian noise on the
SecAgg+ lattice costs nothing (0.690 vs 0.688 at ε = 1). A membership-inference audit
(`dp_audit.csv`) gives attack AUC 0.533 and empirical ε ≥ 0.58 for the no-DP protocol,
and ≤ 0.502 / ≈ 0 for every DP setting.

DP rows use one configuration fixed for all ε (800 random-structure trees, depth 4, η 0.1, 16 bins, λ 10; player clips 5/10), the public league FG% as initial prediction and no per-tree validation, so every release is charged. Sources: `results/federated/eval_summary.csv`, `hist_dp_fixed_summary.csv` (headline + clip / tree-budget sensitivity), `hist_dp_frontier.csv` (per-ε tuned sweep, unaccounted upper bound).

## Known issues / caveats

- **Legacy spacing feature.** `spatial.get_spacing_area` now returns true hull areas (`ConvexHull.volume`), but the shipped `shot_features_valid2.csv` was extracted with the old perimeter-based version (2-D `ConvexHull.area`), so its `ratio_off_def_hull` column is a perimeter ratio. Re-extract if the area semantics matter; the trained models are consistent with the shipped CSV.
- **`compute_poe.py` must run before `compute_def_poe.py`/`correlate_poe.py`** — the latter two read `results/poe_per_shot.csv` / `results/poe_leaderboard.csv`.
