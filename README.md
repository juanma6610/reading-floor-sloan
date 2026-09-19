<h1 align="center">Reading the Floor</h1>
<p align="center"><b>A portable, calibrated, <i>federated</i> shot quality model for the NBA from raw optical player tracking.</b></p>

<p align="center">
  <img src="assets/hero_possession.gif" width="94%" alt="SportVU tracking reconstruction beside the broadcast — Curry catch-and-shoot 3, GSW @ CLE">
  <br><sub>Raw optical tracking reconstructed (left) beside the broadcast footage (right) — Curry's catch and shoot 3, GSW @ CLE.</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-gradient%20boosting-EC4E20">
  <img src="https://img.shields.io/badge/Flower-federated%20learning-30B6EF">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/NBA-C8102E?labelColor=1D428A" alt="NBA SportVU 25Hz">
  <img src="https://img.shields.io/badge/MSc%20thesis-KU%20Leuven-1D428A">
</p>

---

## Summary

Converted raw optical tracking and Play by Play logs into a **calibrated model** that outputs the probability of converting the shot, using only information available *before the ball is released* and then showed that model can be **trained across all 30 teams without any team sharing its raw data** (federated learning) **at no measurable accuracy cost**, and with **formal differential privacy** at a small one.

The calibrated probability powers a suite of analytics: a **Points Over Expectation (POE)** rating for shooters and defenders, per zone and archetype matchup breakdowns, and five man **lineup** evaluation.

> Advanced MSc Artificial Intelligence thesis (Big Data Analytics), KU Leuven — *Juan Manuel Oliver*.

**Highlights**
- 🎯 **Well-calibrated** shot model: Brier **0.204**, log-loss **0.591**, ROC-AUC **0.732** on a *game-disjoint* test set (95 held-out games, 14,844 shots), far above the ~0.60 AUC of a distance-only model.
- 🔒 **Federated across 30 teams** with Flower: a histogram protocol in which all teams grow shared trees from summed gradient histograms matches centralized training (Brier 0.2046 vs 0.2050), with optional **SecAgg+** and **distributed differential privacy** (ε = 1: AUC 0.693). Compared against tree bagging (+6.4% Brier), 5 seeds, game-level bootstrap CIs.
- 🧠 **Mostly portable feature set** — release geometry, defender pressure, shooter/defender kinematics, spacing, tempo, and behavioural **archetypes** from a Gaussian Mixture Model, no dependency on NBA specific player IDs; plus **shot-type descriptors** (layup, dunk, floater, pull-up, …) parsed from the play-by-play text — the one non-tracking input, worth +0.06 AUC.
- 📊 **Applications**: POE leaderboards (shooters & defenders), per zone calibration, archetype matchup heatmaps, lineup POE, and spatial shot charts, ability to train model while keeping data private.
- 🧪 End-to-end, reproducible pipeline from raw `.7z` tracking archives → features → model → federated experiments → thesis figures.

---

## The problem

Some of the most valuable data in sports: practice tracking, biometrics, scheme indicators, is exactly the data teams most want to keep private. That creates a tension: build cutting edge models, *or* respect data governance. This project resolves it by pairing a **deliberately portable shot quality model** with a **cross-silo federated protocol**, so a league scale model can be trained **without any franchise surrendering its raw data**.

## How it works

<p align="center"><img src="assets/pipeline_architecture.png" width="85%" alt="End-to-end pipeline architecture"></p>

Every shot is reduced to **the exact game state at the moment of release**. The hardest part is recovering that release frame from the tracking stream, the PBP timestamp trails the true release by 1–2 seconds, so I detect the ball's arc apex and walk back to the frame where it leaves the shooter's hands.

<p align="center">
  <img src="assets/release_recovery_sidebyside.gif" width="92%" alt="Release-frame recovery synced with broadcast footage of Klay Thompson's catch-and-shoot 3">
  <br><sub><b>Release frame recovery, synced to the broadcast.</b> The algorithm finds the ball's arc apex and walks back to the release frame — here it lines up in real time with Klay Thompson's catch and shoot 3 (GSW @ HOU).</sub>
</p>


The feature set spans six portable families: **shot geometry**, **defender pressure** (distance, angle, closeout time, tight-contest counts), **shooter & defender kinematics**, **release mechanics** (height, speed, angle), **possession tempo** (shot clock, touch time, catch-and-shoot), **floor spacing**, and **soft player archetypes** from a GMM.

<p align="center">
  <img src="assets/shot_geometry_features.png" width="82%" alt="Shot geometry & kinematic features">
  <br><sub><b>Shot geometry &amp; kinematics</b> decomposed parallel / perpendicular to the shot line.</sub>
</p>

<p align="center">
  <img src="assets/touch_time_sidebyside.gif" width="92%" alt="Touch-time recovery synced with broadcast footage of Harden's pull-up 3">
  <br><sub><b>Touch time recovery, synced to the broadcast.</b> Touch time is how long the shooter holds the ball before shooting — here 5.24 s of Harden dribbling into a pull-up 3, catch to release, aligned in real time (GSW @ HOU).</sub>
</p>

Spacing and pressure are dynamic, a kinematic **space control** model turns positions and velocities into who would reach each patch of floor first, revealing how a possession opens (and closes) the shooter's window:

<p align="center">
  <img src="assets/possession_spacing.png" width="80%" alt="Animated space-control (time-to-control) heatmap for Curry's catch-and-shoot 3">
  <br><sub><b>Space control over a possession</b> — blue = offense would arrive first, red = defense. Curry (yellow ring) springs open just before the catch.</sub>
</p>

## Results

A gradient-boosted model trained with **game-disjoint** splits, no game ever spans train and test and evaluated on **probability quality**, not threshold accuracy (because everything downstream integrates the probability, not the label).

| Model | Brier ↓ | Log-loss ↓ | ROC-AUC ↑ |
|---|---|---|---|
| Constant (base rate) | 0.2472 | 0.6875 | 0.500 |
| Distance-only logistic | 0.2411 | 0.6753 | 0.595 |
| **XGBoost (full features)** | **0.2042** | **0.5914** | **0.732** |

<table>
<tr>
<td width="50%"><img src="assets/calibration_diagram.png" alt="Reliability diagram"></td>
<td width="50%"><img src="assets/shap_beeswarm.png" alt="Feature attributions"></td>
</tr>
<tr>
<td align="center"><sub><b>Calibration</b> — predicted probabilities track empirical make rates.</sub></td>
<td align="center"><sub><b>What the model uses</b> — distance, then defender pressure, mechanics, archetypes.</sub></td>
</tr>
</table>

### Federated learning — no accuracy cost, and real privacy on top

The 30 teams are the silos: no raw shot ever leaves its team. How the teams are combined decides both accuracy and privacy.

| Protocol (team silos, 5 seeds) | Brier ↓ | ROC-AUC ↑ | vs centralized (95% CI) |
|---|---|---|---|
| Centralized, same hyperparameters | 0.2050 | 0.729 | — |
| **Histogram protocol** — shared trees from summed gradient histograms | **0.2046** | **0.731** | **−0.2% [−0.5, 0.0]** |
| Tree bagging (`FedXgbBagging`) — each team grows its own trees | 0.2181 | 0.685 | +6.4% [+5.7, +7.0] |
| One team alone | 0.2249 | 0.659 | — |

- **Tree bagging leaks.** From a single round-1 tree, an honest-but-curious server recovers each team's *exact* shot and make counts in every region the tree carves out (591/591 regions across 30 teams), even through leaf-noise "DP" (`leakage_attack.py`).
- **The histogram protocol** only releases per-bin sums on a public bin grid. Under **SecAgg+** the server sees only the total over all 30 teams; without noise the trees are identical to centralized XGBoost.
- **Distributed differential privacy** (each team adds 1/√30 of the Gaussian noise; Rényi-DP accounting, δ = 10⁻⁵, one shot protected): at **ε = 1** the model still reaches AUC **0.693** / Brier 0.2167, better than tree bagging *without* any privacy; at ε = 4, AUC 0.707.

<p align="center"><img src="assets/federated_convergence.png" width="80%" alt="Federated convergence vs centralized baseline"></p>

## Applications — what the calibrated probability unlocks

With a trustworthy P(make), a shot's value over an average shooter in the same situation is simply `POE = value × (outcome − P(make))`, computed **out of fold** so no player is flattered by the model training on their own shots.

<p align="center">
  <img src="assets/per_zone_poe.png" width="85%" alt="Per-zone POE decomposition">
  <br><sub><b>Per-zone POE</b> — where each player creates or sheds expected points.</sub>
</p>

<p align="center">
  <img src="assets/lineup_poe_leaderboard.png" width="92%" alt="Lineup POE leaderboards">
  <br><sub><b>Five-man lineup POE</b> — best and worst offensive &amp; defensive units.</sub>
</p>

<table>
<tr>
<td width="50%"><img src="assets/shot_heatmap_curry_lbj.png" alt="Spatial shot charts"></td>
<td width="50%"><img src="assets/matchup_heatmap.png" alt="Archetype matchup heatmap"></td>
</tr>
<tr>
<td align="center"><sub><b>Spatial shot charts</b> different players have different shot regimes.</sub></td>
<td align="center"><sub><b>Archetype matchups</b> — which styles beat which.</sub></td>
</tr>
</table>


## Repository tour

```
src/                 Feature extraction, model training/tuning, POE, visualization
  shot_features.py     Release-frame recovery + 37-feature extraction per shot
  train_xgboost.py     Game-disjoint calibrated model + evaluation
  compute_poe.py       Out-of-fold Points Over Expectation
  federated/           Flower app: histogram protocol (+SecAgg+, DP) and bagging/cyclic, leakage attack, evaluation
clusters/            R project: GMM player-archetype clustering
figures_thesis/      Scripts that regenerate every thesis figure
docs/                Rerun runbook + dataset documentation
assets/              Figures used in this README
```


## Dataset

The engineered shot features table (≈98k shots × 48 columns) is published as a standalone dataset. The full column dictionary and module level details live in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.
📊 Kaggle: https://www.kaggle.com/datasets/juanmaoliver/shot-features/data


## About

**Juan Manuel Oliver** — MSc Artificial Intelligence (Big Data Analytics), KU Leuven.
Thesis: *Reading the Floor — A Portable Federated Shot Quality Model from Optical Tracking.*

- 🔗 LinkedIn: https://www.linkedin.com/in/juanma-oliver
- 📄 Thesis PDF: 
- 📊 Kaggle: [JuanmaOliver](https://www.kaggle.com/juanmaoliver)

<sub>Built on publicly posted SportVU tracking and NBA play by play data, for research and educational use. Please credit the original data sources.</sub>
