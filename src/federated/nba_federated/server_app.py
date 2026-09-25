"""
server_app.py — Flower ServerApp for federated NBA shot prediction.

`protocol = "histogram"` hands the run to hist_server.py (shared trees built from
summed gradient histograms, optional SecAgg+ and distributed DP). Otherwise
(`protocol = "bagging"`) it runs tree bagging with one of two server-side
aggregation strategies, switchable via the `aggregation` config key in
pyproject.toml ([tool.flwr.app.config]):

  "bagging"  — FedXgbBagging. All clients train each round; the server
               concatenates each client's new tree(s) into the global ensemble.
               PRIMARY EXPERIMENT.

  "cyclic"   — FedXgbCyclic. One client trains per round in a fixed round-robin
               (every client trains exactly once per 30 rounds); its full
               updated booster becomes the next round's global model.
               ABLATION — no within-round tree conflicts, so it serves as a
               drift-free reference point against bagging on Non-IID data.

The two modes write to DIFFERENT output files so you can keep both runs
side-by-side for thesis comparison (in <project>/results/federated/):
   federated_<mode>_metrics.csv           (global-test metrics per round, for curves only)
   xgb_federated_<mode>_model.json        (final round)

No checkpoint is selected here. Every round's global model is a prefix of the
final booster (bagging: 1 iteration = 30 parallel trees per round; cyclic: 1
tree per round), so the reporting round is chosen AFTER training from federated
validation loss, never from the test set — see evaluate_federated.py.
"""

from logging import INFO, WARNING
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common import Context, FitIns, Parameters
from flwr.common.logger import log
from flwr.server import LegacyContext, ServerApp, ServerConfig, ServerAppComponents
from flwr.server.workflow import DefaultWorkflow
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss


# ──────────────────────────────────────────────────────────────
# Strategy adapter: give FedXgbCyclic the same centralised-eval API
# that FedXgbBagging exposes (`evaluate_function` on raw Parameters)
# ──────────────────────────────────────────────────────────────

class FedXgbCyclicWithCentralEval(FedXgbCyclic):
    """
    FedXgbCyclic that supports a centralised evaluation function operating
    directly on Flower `Parameters` (XGBoost JSON bytes), matching the API
    used by FedXgbBagging.

    Why this exists: FedXgbCyclic inherits from FedAvg, whose `evaluate_fn`
    is invoked AFTER `parameters_to_ndarrays(parameters)` — which assumes the
    tensors are numpy-serialised arrays. Our XGBoost models are stored as raw
    JSON bytes, so the default FedAvg path would crash. Overriding `evaluate`
    here bypasses the ndarrays conversion and passes the raw Parameters to
    the user-provided eval fn, exactly the way FedXgbBagging does.
    """

    def __init__(self, *args, evaluate_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluate_function = evaluate_function

    def configure_fit(self, server_round, parameters, client_manager):
        """Deterministic round-robin. The stock FedXgbCyclic indexes into a
        freshly *shuffled* client sample each round, so it actually picks a
        random client with replacement and teams train unequally often."""
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        client_manager.wait_for(self.min_available_clients)
        clients = [client_manager.all()[cid] for cid in sorted(client_manager.all())]
        chosen = clients[(server_round - 1) % len(clients)]
        log(INFO, f"[Server/cyclic] Round {server_round}: client {chosen.cid}")
        return [(chosen, FitIns(parameters, config))]

    def evaluate(self, server_round, parameters):
        if self._evaluate_function is None:
            return None
        result = self._evaluate_function(server_round, parameters, {})
        if result is None:
            return None
        loss, metrics = result
        return loss, metrics



from nba_federated.task import load_global_test_set, get_global_make_rate

# ──────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────
from nba_federated.task import OUTPUT_DIR  # noqa: E402


def _paths_for(mode: str):
    """Return (final_model, metrics_csv) paths for the given mode."""
    return (
        OUTPUT_DIR / f"xgb_federated_{mode}_model.json",
        OUTPUT_DIR / f"federated_{mode}_metrics.csv",
    )


# ──────────────────────────────────────────────────────────────
# Per-round config sent to every client
# ──────────────────────────────────────────────────────────────

def make_on_fit_config_fn(run_config: dict, base_score: float, aggregation: str):
    """
    Build the dict of params shipped with FitIns to every client each round.
    Without this, clients would silently fall back to their hardcoded defaults.
    """
    defaults = {
        "params.objective":         "binary:logistic",
        "params.eval_metric":       "logloss",
        "params.tree_method":       "hist",
        "params.max_depth":         5,
        "params.eta":               0.05,
        "params.subsample":         0.8,
        "params.colsample_bytree":  0.8,
        "params.min_child_weight":  10,
        "params.reg_lambda":        2.0,
        "params.base_score":        base_score,
        "local-epochs":             1,
        "aggregation":              aggregation,
        "seed":                     42,
        # Differential privacy (off by default): total epsilon budget and the
        # leaf clip bound. eps <= 0 disables DP on the client side.
        "dp-epsilon":               0.0,
        "dp-clip":                  0.3,
        "dp-delta":                 1e-5,
    }
    cfg = {**defaults}
    for k, v in run_config.items():
        if k in defaults:
            cfg[k] = v
    # The aggregation flag is decided server-side; never let run_config override
    # it for clients (would create a mismatch with the strategy actually used).
    cfg["aggregation"] = aggregation
    # Clients need their own RELEASE count to compose the per-round epsilon — which is not
    # the server's round count under cyclic aggregation, where exactly one client trains per
    # round and each is visited once every num-clients rounds. Charging all 1500 rounds to
    # every client would over-noise by ~30x.
    rounds = int(run_config.get("num-server-rounds", 50))
    if aggregation == "cyclic":
        rounds = max(1, rounds // int(run_config.get("num-clients", 30)))
    cfg["dp-num-rounds"] = rounds

    def on_fit_config(server_round: int) -> dict:
        return {**cfg, "server-round": server_round}

    return on_fit_config


# ──────────────────────────────────────────────────────────────
# Global evaluation function (called by server after each round)
# ──────────────────────────────────────────────────────────────

def make_evaluate_fn(data_path: str | None, mode: str):
    """
    Returns a centralised evaluation function the strategy calls after each
    aggregation round. It LOGS global-test metrics for convergence curves only;
    nothing here is used to pick the reported model. Writes mode-specific files
    so bagging vs cyclic runs don't overwrite each other.
    """
    latest_path, metrics_csv = _paths_for(mode)
    test_dmatrix, y_test = load_global_test_set(data_path=data_path)
    metrics_log = []

    def evaluate_fn(server_round: int, parameters: Parameters, config: dict):
        if not parameters.tensors or parameters.tensors[0] == b"":
            log(WARNING, f"[Server/{mode}] Round {server_round}: empty params, skipping eval.")
            return 0.0, {}

        booster = xgb.Booster()
        booster.load_model(bytearray(parameters.tensors[0]))

        y_probs = booster.predict(test_dmatrix)
        y_preds = (y_probs > 0.5).astype(int)

        auc      = float(roc_auc_score(y_test, y_probs))
        accuracy = float(accuracy_score(y_test, y_preds))
        brier    = float(brier_score_loss(y_test, y_probs))
        n_trees  = booster.num_boosted_rounds()
        loss = float(-np.mean(
            y_test * np.log(y_probs + 1e-7) +
            (1 - y_test) * np.log(1 - y_probs + 1e-7)
        ))

        log(INFO,
            f"[Server/{mode}] Round {server_round:>3d} | "
            f"Trees={n_trees:>5d} | AUC={auc:.4f} | "
            f"Acc={accuracy:.4f} | Brier={brier:.4f}")

        metrics_log.append({
            "round":    server_round,
            "n_trees":  n_trees,
            "auc":      auc,
            "accuracy": accuracy,
            "brier":    brier,
            "logloss":  loss,
        })

        booster.save_model(str(latest_path))
        pd.DataFrame(metrics_log).to_csv(str(metrics_csv), index=False)

        return loss, {"auc": auc, "accuracy": accuracy, "brier": brier}

    return evaluate_fn


# ──────────────────────────────────────────────────────────────
# ServerApp factory
# ──────────────────────────────────────────────────────────────

def server_fn(context: Context) -> ServerAppComponents:
    """Configure and return the Flower server components."""
    num_rounds  = int(context.run_config.get("num-server-rounds", 50))
    num_clients = int(context.run_config.get("num-clients", 30))
    data_path   = context.run_config.get("data-path", None)
    aggregation = str(context.run_config.get("aggregation", "bagging")).lower()

    if aggregation not in {"bagging", "cyclic"}:
        log(WARNING, f"[Server] Unknown aggregation '{aggregation}', falling back to 'bagging'.")
        aggregation = "bagging"

    base_score = get_global_make_rate(data_path)

    log(INFO, f"[Server] Starting federated XGBoost training")
    log(INFO, f"[Server] Mode={aggregation} | Rounds={num_rounds} | "
              f"Clients={num_clients} | base_score={base_score:.4f}")

    on_fit_config_fn = make_on_fit_config_fn(context.run_config, base_score, aggregation)
    evaluate_fn      = make_evaluate_fn(data_path, aggregation)

    if aggregation == "cyclic":
        # 1 client per round in fixed round-robin order (see configure_fit
        # override). The selected client's full booster becomes the next
        # round's global model.
        # We use a thin subclass that lets us pass an XGBoost-aware central
        # evaluator (the stock FedXgbCyclic inherits FedAvg's `evaluate_fn`,
        # which incorrectly tries to convert our JSON-bytes Parameters to
        # numpy ndarrays). See FedXgbCyclicWithCentralEval above.
        strategy = FedXgbCyclicWithCentralEval(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=1,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            on_fit_config_fn=on_fit_config_fn,
            evaluate_function=evaluate_fn,
        )
    else:
        # FedXgbBagging: every client trains every round; their new trees are
        # concatenated into the global ensemble.
        strategy = FedXgbBagging(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            on_fit_config_fn=on_fit_config_fn,
            evaluate_function=evaluate_fn,
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp entry point. Main-style so the histogram protocol can run under
# the SecAgg+ workflow; the bagging/cyclic strategies run through the same legacy
# workflow Flower uses for `ServerApp(server_fn=...)`.
app = ServerApp()


@app.main()
def main(grid, context: Context) -> None:
    if str(context.run_config.get("protocol", "bagging")) == "histogram":
        from nba_federated import hist_server
        hist_server.run(grid, context)
        return
    components = server_fn(context)
    legacy = LegacyContext(context=context, config=components.config, strategy=components.strategy)
    DefaultWorkflow()(grid, legacy)
