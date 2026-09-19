"""
visualize_possessions.py — look at the possessions the dataset builder produced.

Reconstructs court positions straight from the compact dataset (X[...,0]*94,
X[...,1]*50 — no raw tracking needed) and renders the possessions of a game two
ways:

  * INTERACTIVE HTML (default): a self-contained page with a dropdown of every
    possession (index, value R, terminal type), play/pause, a frame slider, and a
    court canvas. Offense = blue, defense = red, ball = orange, and the current
    ball-handler gets a yellow ring — so you can watch the play develop and see
    who has the ball at each frame. -> results/epv/viz/<gid>.html
  * PNG contact sheet: a grid of possessions drawn as player trajectories (start
    dot -> end), titled with R + terminal. Good for eyeballing many at once.
    -> results/epv/viz/<gid>_contact.png
  * EPV OVERLAY (--epv): under the court, the trained causal model's per-frame
    value curve V(t) for the selected possession, with each frame shaded by
    dEPV(t)=V(t+1)-V(t) — green where the play added value, red where it lost it.
    A good drive walks the curve up; a turnover drops it to ~0. The curve is
    drawn to the same frame index as the animation, so scrubbing moves a marker
    along it. `--epv-png` writes a static figure of a few example curves.

EPV values come, in order of preference, from
  1. results/epv/epv_crossfit_V.npy   (epv_actions.py K-fold cross-fit — every
     possession scored by a model that never saw its game), else
  2. results/epv/epv_causal_full_seed0.pt (a trained checkpoint; the header says
     whether this game was in that model's TRAINING games, in which case the
     curve is in-sample and only illustrative).

Node order per frame: [0]=ball, [1..5]=offense, [6..10]=defense.

Run:
  python src/epv/visualize_possessions.py --game 21500485            # HTML
  python src/epv/visualize_possessions.py --game 21500485 --epv      # + EPV curve
  python src/epv/visualize_possessions.py --game 21500485 --png      # + contact sheet
  python src/epv/visualize_possessions.py --game 21500485 --epv-png  # example curves
"""
from __future__ import annotations
import os, glob, json, argparse
import numpy as np

NPZ = "results/epv/epv_possessions.npz"
SHARD_DIR = "results/epv/shards"
VIZ_DIR = "results/epv/viz"
CROSSFIT_V = "results/epv/epv_crossfit_V.npy"
CKPT = "results/epv/epv_causal_full_seed0.pt"
TERM_NAME = {0: "made FG", 1: "made FT", 2: "miss+DReb", 3: "turnover",
             4: "reb end", 5: "period end", 6: "other"}


def load_game(gid):
    """Return the dataset arrays for one game (from shard if present, else npz)."""
    shard = os.path.join(SHARD_DIR, f"{gid}.npz")
    if os.path.exists(shard):
        d = dict(np.load(shard))
    else:
        d = dict(np.load(NPZ))
        sel = d["games"] == gid
        d = {k: v[sel] for k, v in d.items()}
    if len(d["R"]) == 0:
        raise SystemExit(f"No possessions for game {gid}. Available: "
                         f"{sorted(set((np.load(NPZ)['games']).tolist())) if os.path.exists(NPZ) else '(build dataset first)'}")
    return d


# ──────────────────────────────────────────────────────────────────────────
# EPV values for a game (cross-fit cache first, then a trained checkpoint)
# ──────────────────────────────────────────────────────────────────────────
def epv_values(gid, d):
    """Return (V:[n_poss, KMAX], source_label) or (None, reason)."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    n, K = len(d["R"]), d["mask"].shape[1]

    if os.path.exists(CROSSFIT_V):
        try:
            from epv_data import load_epv
            full = load_epv(verbose=False, in_ram=False)
            rows = np.flatnonzero(np.asarray(full["games"]) == gid)
            V = np.load(CROSSFIT_V)
            if len(rows) == n and not np.isnan(V[rows]).all():
                return V[rows], "K-fold cross-fit (out-of-sample)"
            print(f"  (cross-fit cache has {len(rows)} rows for {gid}, dataset slice has {n}"
                  f" — falling back to the checkpoint)")
        except Exception as e:
            print(f"  (cross-fit cache unusable: {e})")

    if not os.path.exists(CKPT):
        return None, f"no {CROSSFIT_V} and no {CKPT} — train first"
    try:
        import torch
        import gnn_epv_causal as G
        ck = torch.load(CKPT, map_location=G.DEVICE, weights_only=False)
        model = G.GNNEPVCausal(ck["fin"], gru_h=ck.get("gru_h", G.GRU_H),
                               mode=ck.get("mode", "full")).to(G.DEVICE)
        model.load_state_dict(ck["state_dict"]); model.eval()
        with torch.no_grad():
            X = torch.tensor(d["X"], dtype=torch.float32, device=G.DEVICE)
            m = torch.tensor(d["mask"], dtype=torch.float32, device=G.DEVICE)
            V = model(X, m).cpu().numpy()
        seen = ""
        try:                                          # was this game in training?
            from epv_data import load_epv, game_splits
            full = load_epv(verbose=False, in_ram=False)
            tr, va, te = game_splits(np.asarray(full["games"]), full["idx"],
                                     seed=int(ck.get("seed", 0)))
            g = np.asarray(full["games"])
            seen = (" [held-out game: out-of-sample]" if gid in set(g[te].tolist())
                    else " [WARNING: this game was in TRAINING — curve is in-sample]")
        except Exception:
            pass
        return V, f"checkpoint {os.path.basename(CKPT)}{seen}"
    except Exception as e:
        return None, f"could not score with {CKPT}: {e}"


# ──────────────────────────────────────────────────────────────────────────
# Interactive HTML
# ──────────────────────────────────────────────────────────────────────────
def export_html(gid, d, V=None, vsrc=""):
    os.makedirs(VIZ_DIR, exist_ok=True)
    X, klen, bh, R, term = d["X"], d["klen"], d["bh"], d["R"], d["term"]
    poss = []
    for i in range(len(R)):
        k = int(klen[i])
        xy = X[i, :k, :, :2].copy()
        xy[..., 0] *= 94.0; xy[..., 1] *= 50.0                 # de-normalize to feet
        rec = {
            "k": k,
            "xy": np.round(xy, 1).tolist(),                    # [k][11][2]
            "bh": bh[i, :k].astype(int).tolist(),
            "R": int(R[i]),
            "term": TERM_NAME.get(int(term[i]), "?"),
        }
        if V is not None:
            rec["v"] = np.round(V[i, :k], 4).tolist()          # per-frame EPV
        poss.append(rec)
    payload = json.dumps({"gid": int(gid), "poss": poss, "vsrc": vsrc},
                         separators=(",", ":"))
    html = _HTML_TEMPLATE.replace("__DATA__", payload)
    out = os.path.join(VIZ_DIR, f"{gid}.html")
    open(out, "w").write(html)
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out}  ({len(poss)} possessions, {kb:.0f} KB)")
    return out


_HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Possession viewer</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:12px;color:#222}
 #wrap{max-width:900px;margin:auto}
 .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}
 select,button{font-size:14px;padding:4px 8px}
 #meta{font-weight:600}
 canvas{border:1px solid #ccc;background:#f7efe0;border-radius:6px;width:100%;height:auto}
 #v{background:#fff}
 .legend span{display:inline-block;margin-right:14px}
 .dot{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:4px}
</style></head><body><div id="wrap">
<h3>Game <span id="gid"></span> — possession viewer</h3>
<div class="row">
  <label>Possession <select id="sel"></select></label>
  <button id="play">▶ Play</button>
  <button id="restart">⟲ Restart</button>
  <label>Speed <input id="speed" type="range" min="1" max="8" value="3"></label>
</div>
<div class="row"><span id="meta"></span></div>
<canvas id="c" width="940" height="500"></canvas>
<div class="row"><input id="scrub" type="range" min="0" value="0" style="flex:1"><span id="frame"></span></div>
<div id="epvbox" style="display:none">
  <h4 style="margin:14px 0 2px">Expected possession value V(t) — shaded by ΔEPV per frame</h4>
  <div class="row"><span id="epvmeta"></span></div>
  <canvas id="v" width="940" height="210"></canvas>
  <div class="row legend">
    <span><i class="dot" style="background:#9ec5f4"></i>ΔEPV &gt; 0 (value added)</span>
    <span><i class="dot" style="background:#f2b4b4"></i>ΔEPV &lt; 0 (value lost)</span>
    <span><i class="dot" style="background:#0b0b0b"></i>V(t), pts</span>
    <span style="color:#898781">dashed = league PPP 1.05</span>
  </div>
</div>
<div class="row legend">
  <span><i class="dot" style="background:#1D428A"></i>offense</span>
  <span><i class="dot" style="background:#C8102E"></i>defense</span>
  <span><i class="dot" style="background:#E8850C"></i>ball</span>
  <span><i class="dot" style="background:#f7efe0;border:3px solid #FFD200"></i>ball-handler</span>
</div>
</div>
<script>
const DATA=__DATA__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const W=94,H=50,SX=cv.width/W,SY=cv.height/H;
const sel=document.getElementById('sel'),meta=document.getElementById('meta');
const scrub=document.getElementById('scrub'),frameLbl=document.getElementById('frame');
const playBtn=document.getElementById('play'),speed=document.getElementById('speed');
document.getElementById('gid').textContent=DATA.gid;
DATA.poss.forEach((p,i)=>{const o=document.createElement('option');
  o.value=i;o.textContent=`#${i}  R=${p.R}  ${p.term}`;sel.appendChild(o);});
let cur=0,fr=0,playing=false,acc=0;
const vc=document.getElementById('v'),vx=vc.getContext('2d');
const epvbox=document.getElementById('epvbox'),epvmeta=document.getElementById('epvmeta');
const PPP=1.05, VPAD={l:46,r:12,t:14,b:26};
function vScale(p){
  const lo=Math.min(PPP,Math.min.apply(null,p.v))-0.15;
  const hi=Math.max(PPP,Math.max.apply(null,p.v))+0.15;
  const w=vc.width-VPAD.l-VPAD.r, h=vc.height-VPAD.t-VPAD.b;
  return {x:t=>VPAD.l+(p.k<2?0:t/(p.k-1))*w, y:v=>VPAD.t+h*(1-(v-lo)/(hi-lo)),
          lo:lo, hi:hi, w:w, h:h, dx:p.k<2?w:w/(p.k-1)};
}
function drawV(){
  const p=DATA.poss[cur];
  if(!p.v){epvbox.style.display='none';return;}
  epvbox.style.display='';
  const s=vScale(p), f=Math.min(fr,p.k-1);
  vx.clearRect(0,0,vc.width,vc.height);
  // per-frame ΔEPV shading: the strip covering (t, t+1] is tinted by its sign,
  // opacity proportional to |ΔEPV| (saturating at 0.5 pts).
  for(let t=0;t<p.k-1;t++){
    const dv=p.v[t+1]-p.v[t];
    const a=Math.min(Math.abs(dv)/0.5,1)*0.55;      // saturates at half a point
    vx.fillStyle=(dv>=0?'rgba(42,120,214,':'rgba(227,73,72,')+a.toFixed(3)+')';
    vx.fillRect(s.x(t),VPAD.t,Math.max(s.dx,1),s.h);
  }
  // axes + league-PPP reference
  vx.strokeStyle='#c3c2b7';vx.lineWidth=1;
  vx.beginPath();vx.moveTo(VPAD.l,VPAD.t);vx.lineTo(VPAD.l,VPAD.t+s.h);
  vx.lineTo(VPAD.l+s.w,VPAD.t+s.h);vx.stroke();
  vx.setLineDash([5,4]);vx.strokeStyle='#898781';
  vx.beginPath();vx.moveTo(VPAD.l,s.y(PPP));vx.lineTo(VPAD.l+s.w,s.y(PPP));vx.stroke();
  vx.setLineDash([]);
  vx.fillStyle='#898781';vx.font='11px system-ui,-apple-system,"Segoe UI",sans-serif';
  vx.textAlign='right';
  [s.lo,(s.lo+s.hi)/2,s.hi].forEach(v=>vx.fillText(v.toFixed(2),VPAD.l-6,s.y(v)+4));
  vx.textAlign='left';vx.fillText('frame',VPAD.l,vc.height-8);
  vx.textAlign='right';vx.fillText(`terminal (R=${p.R}, ${p.term})`,VPAD.l+s.w,vc.height-8);
  // V(t)
  vx.strokeStyle='#0b0b0b';vx.lineWidth=2;vx.beginPath();
  for(let t=0;t<p.k;t++){const X0=s.x(t),Y0=s.y(p.v[t]);t?vx.lineTo(X0,Y0):vx.moveTo(X0,Y0);}
  vx.stroke();
  // current frame marker
  vx.strokeStyle='#E8850C';vx.lineWidth=2;
  vx.beginPath();vx.moveTo(s.x(f),VPAD.t);vx.lineTo(s.x(f),VPAD.t+s.h);vx.stroke();
  vx.beginPath();vx.arc(s.x(f),s.y(p.v[f]),4.5,0,2*Math.PI);vx.fillStyle='#E8850C';vx.fill();
  const dv=f<p.k-1?p.v[f+1]-p.v[f]:0;
  epvmeta.innerHTML=`EPV V(t)=<b>${p.v[f].toFixed(3)}</b> pts &nbsp;|&nbsp; `+
    `ΔEPV(t)=<b style="color:${dv>=0?'#1c5cab':'#d03b3b'}">${dv>=0?'+':''}${dv.toFixed(3)}</b>`+
    ` (ball-handler node ${p.bh[f]}) &nbsp;|&nbsp; V(0)=${p.v[0].toFixed(3)} → V(T)=${p.v[p.k-1].toFixed(3)}`+
    ` &nbsp;|&nbsp; <span style="color:#888">values: ${DATA.vsrc||'n/a'}</span>`;
}
function X(x){return x*SX} function Y(y){return cv.height-y*SY}
function arc(cx,cy,r,a0,a1){ctx.beginPath();ctx.arc(X(cx),Y(cy),r*SX,a0,a1);ctx.stroke();}
function court(){
 ctx.clearRect(0,0,cv.width,cv.height);ctx.strokeStyle='#b79b6b';ctx.lineWidth=2;
 ctx.strokeRect(X(0),Y(50),94*SX,50*SY);
 ctx.beginPath();ctx.moveTo(X(47),Y(0));ctx.lineTo(X(47),Y(50));ctx.stroke();
 arc(47,25,6,0,2*Math.PI);
 [[5.25,25],[88.75,25]].forEach(h=>{arc(h[0],h[1],0.75,0,2*Math.PI);
   ctx.beginPath();ctx.arc(X(h[0]),Y(h[1]),23.75*SX,0,2*Math.PI);ctx.stroke();});
}
function draw(){
 const p=DATA.poss[cur];court();
 const f=Math.min(fr,p.k-1);const pts=p.xy[f];const bh=p.bh[f];
 // faint trail of the ball
 ctx.strokeStyle='rgba(232,133,12,.35)';ctx.lineWidth=2;ctx.beginPath();
 for(let t=0;t<=f;t++){const b=p.xy[t][0];t?ctx.lineTo(X(b[0]),Y(b[1])):ctx.moveTo(X(b[0]),Y(b[1]));}
 ctx.stroke();
 for(let n=0;n<11;n++){
   const [x,y]=pts[n];const off=(n>=1&&n<=5);
   const col=n===0?'#E8850C':(off?'#1D428A':'#C8102E');
   const r=n===0?6:10;
   if(n===bh){ctx.beginPath();ctx.arc(X(x),Y(y),r+5,0,2*Math.PI);ctx.strokeStyle='#FFD200';ctx.lineWidth=3;ctx.stroke();}
   ctx.beginPath();ctx.arc(X(x),Y(y),r,0,2*Math.PI);ctx.fillStyle=col;ctx.fill();
 }
 frameLbl.textContent=`frame ${f+1}/${p.k}`;scrub.value=f;
 meta.textContent=`Possession #${cur} — value R=${p.R} — terminal: ${p.term} — ${p.k} frames`;
 drawV();
}
function loadPoss(i){cur=i;fr=0;scrub.max=DATA.poss[i].k-1;draw();}
function tick(){ if(!playing)return;
 acc++; if(acc>=(9-speed.value)){acc=0; fr++; if(fr>=DATA.poss[cur].k){playing=false;playBtn.textContent='▶ Play';fr=DATA.poss[cur].k-1;} draw();}
 requestAnimationFrame(tick);
}
sel.onchange=e=>loadPoss(+e.target.value);
scrub.oninput=e=>{fr=+e.target.value;draw();};
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'⏸ Pause':'▶ Play';if(playing){if(fr>=DATA.poss[cur].k-1)fr=0;tick();}};
document.getElementById('restart').onclick=()=>{fr=0;draw();};
loadPoss(0);
</script></body></html>"""


# ──────────────────────────────────────────────────────────────────────────
# PNG contact sheet (trajectories)
# ──────────────────────────────────────────────────────────────────────────
def draw_court(ax):
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((0, 0), 94, 50, fill=False, ec="#b79b6b", lw=1.5))
    ax.plot([47, 47], [0, 50], color="#b79b6b", lw=1)
    for hx in (5.25, 88.75):
        ax.add_patch(Circle((hx, 25), 0.75, fill=False, ec="#b79b6b"))
        ax.add_patch(Circle((hx, 25), 23.75, fill=False, ec="#b79b6b", lw=0.8))
    ax.set_xlim(-1, 95); ax.set_ylim(-1, 51); ax.set_aspect("equal"); ax.axis("off")


def export_png(gid, d, n=12):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(VIZ_DIR, exist_ok=True)
    X, klen, R, term = d["X"], d["klen"], d["R"], d["term"]
    n = min(n, len(R)); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6))
    for i, ax in enumerate(axes.flat):
        if i >= n: ax.axis("off"); continue
        k = int(klen[i]); xy = X[i, :k, :, :2].copy(); xy[..., 0] *= 94; xy[..., 1] *= 50
        draw_court(ax)
        for node in range(11):
            off = 1 <= node <= 5
            col = "#E8850C" if node == 0 else ("#1D428A" if off else "#C8102E")
            ax.plot(xy[:, node, 0], xy[:, node, 1], color=col, lw=1.2 if node else 2, alpha=.8)
            ax.scatter(xy[-1, node, 0], xy[-1, node, 1], s=22 if node else 30, color=col, zorder=3)
        ax.set_title(f"#{i}  R={int(R[i])}  {TERM_NAME.get(int(term[i]),'?')}", fontsize=9)
    fig.suptitle(f"Game {gid} — first {n} possessions (trajectories; dot = end position)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(VIZ_DIR, f"{gid}_contact.png"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Static EPV-curve figure (the "face validity" plot for docs/epv_gnn.md)
# ──────────────────────────────────────────────────────────────────────────
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"          # diverging pair; gray midpoint = surface
SURF = "#fcfcfb"


def _epv_panel(ax, v, k, R, term, title):
    """One possession's V(t): black line, per-frame ΔEPV wash, PPP reference.
    Sign is carried by the curve's slope and the annotations as well as by hue."""
    t = np.arange(k)
    dv = np.diff(v[:k])
    for i, val in enumerate(dv):
        ax.axvspan(i, i + 1, color=(POS if val >= 0 else NEG),
                   alpha=min(abs(val) / 0.5, 1.0) * 0.55, lw=0)
    ax.axhline(1.05, color=MUTED, ls=(0, (5, 4)), lw=1, zorder=2)
    ax.plot(t, v[:k], color=INK, lw=2, zorder=3, solid_capstyle="round")
    ax.scatter([k - 1], [v[k - 1]], s=42, color=INK, zorder=4)
    ax.annotate(f"R={int(R)}", (k - 1, v[k - 1]), textcoords="offset points",
                xytext=(-6, 10), ha="right", color=INK, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    ax.set_xlim(0, k - 1); ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def export_epv_png(gid, d, V, vsrc, n=6):
    """A grid of example EPV curves, picked to span the terminal types."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(VIZ_DIR, exist_ok=True)
    klen, R, term = d["klen"], d["R"], d["term"]
    # one example per terminal type, preferring the largest swing (most legible)
    picks = []
    for tcode in (0, 2, 3, 1, 4, 5):
        cand = [i for i in range(len(R)) if int(term[i]) == tcode and int(klen[i]) > 10]
        if not cand: continue
        best = max(cand, key=lambda i: abs(V[i, int(klen[i]) - 1] - V[i, 0]))
        picks.append(best)
        if len(picks) == n: break
    cols = 2; rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 2.5),
                             squeeze=False, facecolor=SURF)
    for ax, i in zip(axes.flat, picks):
        k = int(klen[i])
        _epv_panel(ax, V[i], k, R[i], term[i],
                   f"#{i} · {TERM_NAME.get(int(term[i]),'?')} · "
                   f"V(0)={V[i,0]:.2f} → V(T)={V[i,k-1]:.2f}")
    for ax in axes.flat[len(picks):]: ax.axis("off")
    for ax in axes[-1]: ax.set_xlabel("frame within possession", color=MUTED, fontsize=9)
    for row in axes: row[0].set_ylabel("EPV (points)", color=MUTED, fontsize=9)
    fig.suptitle(f"Per-frame EPV — game {gid}    "
                 f"(blue = frame added value, red = lost value; dashed = league PPP 1.05)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"values: {vsrc}", fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(VIZ_DIR, f"{gid}_epv_curves.png")
    fig.savefig(out, dpi=140, facecolor=SURF); plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, required=True, help="game id, e.g. 21500485")
    ap.add_argument("--png", action="store_true", help="also write the PNG contact sheet")
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--n", type=int, default=12, help="possessions in the contact sheet")
    ap.add_argument("--epv", action="store_true",
                    help="overlay the trained model's per-frame V(t) under the court")
    ap.add_argument("--epv-png", action="store_true", dest="epv_png",
                    help="also write a static figure of example EPV curves")
    a = ap.parse_args()
    d = load_game(a.game)
    V, vsrc = (None, "")
    if a.epv or a.epv_png:
        V, vsrc = epv_values(a.game, d)
        print(f"EPV values: {vsrc}" if V is not None else f"EPV unavailable — {vsrc}")
    if not a.png_only:
        export_html(a.game, d, V, vsrc)
    if a.png or a.png_only:
        export_png(a.game, d, a.n)
    if a.epv_png and V is not None:
        export_epv_png(a.game, d, V, vsrc)


if __name__ == "__main__":
    main()
