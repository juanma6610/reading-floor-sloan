"""
Correlate POE with Basketball-Reference advanced metrics (TS%, PER, OBPM, ...),
including partial correlations controlling for USG%.

Run from the project root:
    python src/correlate_poe.py
"""
import pandas as pd, numpy as np, unicodedata, re
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parents[2]  # project root
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

poe = pd.read_csv(BASE / "results" / "poe_leaderboard.csv")
adv = pd.read_csv(BASE / "clusters" / "data" / "ad.csv")
pos = pd.read_csv(BASE / "clusters" / "data" / "pos.csv")

def norm(s):
    if not isinstance(s, str): return s
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[.\-']", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def collapse_bbref(df):
    df = df.copy()
    df["is_tot"] = (df["Team"] == "TOT").astype(int)
    df = df.sort_values(["Player", "is_tot"], ascending=[True, False])
    df = df.drop_duplicates("Player", keep="first").drop(columns=["is_tot"])
    return df

adv = collapse_bbref(adv); pos = collapse_bbref(pos)
adv["_k"] = adv["Player"].map(norm)
pos["_k"] = pos["Player"].map(norm)
poe["_k"] = poe["player_name"].map(norm)

pos_small = pos[["_k","eFG%","PTS","FGA","3PA","FTA"]]
merged = poe.merge(adv, on="_k", how="left", suffixes=("",".adv"))
merged = merged.merge(pos_small, on="_k", how="left", suffixes=("",".pos"))

print("matched advanced:", merged["PER"].notna().sum(), "/", len(poe))
unmatched = merged[merged["PER"].isna()]["player_name"].tolist()
print("still unmatched:", unmatched)

for col in ["PER","TS%","USG%","OBPM","DBPM","BPM","VORP","WS","WS/48","OWS","DWS","eFG%","MP"]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

m = merged.dropna(subset=["PER","TS%","OBPM","BPM","VORP","eFG%"]).copy()
m = m[(m["MP"] >= 500) & (m["total_shots"] >= 100)]
print("usable n:", len(m))

metrics = ["TS%","eFG%","PER","USG%","OBPM","DBPM","BPM","VORP","WS","WS/48"]
rows = []
for metric in metrics:
    for poe_var in ["total_POE","POE_per_100"]:
        x = m[poe_var].values; y = m[metric].values
        pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
        rows.append({"POE_variant": poe_var, "metric": metric, "n": len(x),
                     "pearson_r": round(pr,3), "pearson_p": round(pp,4),
                     "spearman_r": round(sr,3), "spearman_p": round(sp,4)})
corr = pd.DataFrame(rows)
corr.to_csv(OUT_DIR / "poe_vs_advanced_correlations.csv", index=False)
print(corr.to_string(index=False))

def partial(y, x, control):
    X = np.column_stack([np.ones(len(control)), control])
    bx = np.linalg.lstsq(X, x, rcond=None)[0]; rx = x - X @ bx
    by = np.linalg.lstsq(X, y, rcond=None)[0]; ry = y - X @ by
    pr, pp = pearsonr(rx, ry); return pr, pp

ctrl = m["USG%"].values
prows = []
print("\nPartial correlations controlling for USG%:")
for metric in ["TS%","eFG%","OBPM","BPM","VORP","WS/48"]:
    for poe_var in ["total_POE","POE_per_100"]:
        pr, pp = partial(m[metric].values, m[poe_var].values, ctrl)
        print(f"  {poe_var} vs {metric} | USG%:  r={pr:.3f}  p={pp:.4f}")
        prows.append({"POE_variant": poe_var, "metric": metric, "controlled_for": "USG%",
                      "partial_r": round(pr,3), "partial_p": round(pp,4)})
pd.DataFrame(prows).to_csv(OUT_DIR / "poe_partial_correlations.csv", index=False)
m.to_csv(OUT_DIR / "poe_merged_with_advanced.csv", index=False)
print("\nSaved.")
