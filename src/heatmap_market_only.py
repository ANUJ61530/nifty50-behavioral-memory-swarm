import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths & config ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEEDS   = 150
N_BINS    = 20
RANDOM_SEED = 0

# Sweep grids
ETA_LIST = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
M_LIST   = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

BEHAVIORS = ['pushover', 'opportunist', 'traditionalist', 'contrarian', 'curmudgeon']

# ── Data loading ──────────────────────────────────────────────────────
def load_returns(path=DATA_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["q"] = np.sign(df["ret"])
    df.loc[df["q"] == 0, "q"] = 1.0
    df["ma200"] = df["close"].rolling(window=200).mean()
    df["d_ma200"] = df["ma200"].diff()
    df["C"] = np.sign(df["d_ma200"].shift(1))
    df["C"] = df["C"].fillna(1.0)
    df.loc[df["C"] == 0, "C"] = 1.0
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    return df.rename(columns={"timestamp": "date"})

def build_lagged_q(df, lag_days=1):
    df = df.copy()
    df["q_lagged"] = df["q"].shift(lag_days)
    return df

# ── Weights ───────────────────────────────────────────────────────────
def get_agent_weights(personality, M, b_opp=1.5, b_trad=0.7):
    if personality == 'pushover':
        w = np.ones(M)
    elif personality == 'opportunist':
        w = b_opp ** np.arange(M)
    elif personality == 'traditionalist':
        w = b_trad ** np.arange(M) if b_trad < 1.0 else b_trad ** (-np.arange(M))
    elif personality == 'contrarian':
        w = -np.ones(M)
    else:
        w = np.ones(M)
    s = np.sum(np.abs(w))
    return w / s if s > 0 else w

# ── Market-only mixed simulation (p_peer = 0) ────────────────────────
def simulate_market_only(q_market, C_market, M, eta, n_seeds,
                         type_proportions, seed_base=RANDOM_SEED):
    n_days = len(q_market)
    rng = np.random.RandomState(seed_base)

    types = ['pushover', 'opportunist', 'traditionalist', 'contrarian', 'curmudgeon']
    num_per = {}; remaining = n_seeds; tl = []
    for t in types[:-1]:
        n = int(n_seeds * type_proportions.get(t, 0.0))
        num_per[t] = n; remaining -= n
    num_per['curmudgeon'] = max(0, remaining)
    for t, n in num_per.items():
        tl.extend([t] * n)
    ta = np.array(tl)[:n_seeds]
    rng.shuffle(ta)
    is_curm = (ta == 'curmudgeon')

    weights = np.zeros((n_seeds, M))
    for i in range(n_seeds):
        weights[i] = get_agent_weights(ta[i], M)

    buf = np.zeros((n_seeds, M))
    nplus = np.full(n_days, np.nan)

    for t in range(n_days):
        if np.isnan(q_market[t]):
            continue
        pos = t % M
        mbit = np.full(n_seeds, q_market[t])
        flip = rng.rand(n_seeds) < eta
        buf[:, pos] = np.where(flip, -mbit, mbit)
        if t >= M - 1:
            cb = np.roll(buf, -pos - 1, axis=1)
            sig = np.sum(cb * weights, axis=1)
            ops = np.sign(sig)
            ops[ops == 0] = 1.0
            ops[is_curm] = C_market[t]
            nplus[t] = np.mean(ops == 1.0)
    return nplus

# ── Drift fitting ─────────────────────────────────────────────────────
def fit_drift(n_plus, n_bins=N_BINS):
    x  = n_plus[:-1] - 0.5
    dx = n_plus[1:] - n_plus[:-1]
    v = ~np.isnan(x) & ~np.isnan(dx)
    x, dx = x[v], dx[v]
    if len(x) < 30:
        return None
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bi = np.digitize(x, edges) - 1
    bc, bd = [], []
    for b in range(n_bins):
        m = bi == b
        if m.sum() >= 5:
            bc.append(x[m].mean()); bd.append(dx[m].mean())
    bc, bd = np.array(bc), np.array(bd)
    if len(bc) < 4:
        return None
    D = np.column_stack([bc, bc**3])
    co, *_ = np.linalg.lstsq(D, bd, rcond=None)
    return -co[0]   # k value

# ── Load data ─────────────────────────────────────────────────────────
print("Loading data...")
df = load_returns()
df = build_lagged_q(df)
q_market = df["q_lagged"].values
C_market = df["C"].values
print(f"  {len(q_market)} trading days loaded.\n")

# ── Sweep and build heatmaps ──────────────────────────────────────────
for beh in BEHAVIORS:
    print(f"{'='*60}")
    print(f"  Behavior: {beh.upper()}")
    print(f"{'='*60}")

    props = {beh: 1.0}
    k_grid = np.full((len(ETA_LIST), len(M_LIST)), np.nan)

    for i, eta in enumerate(ETA_LIST):
        for j, M in enumerate(M_LIST):
            k = fit_drift(simulate_market_only(
                q_market, C_market, M, eta, N_SEEDS, props))
            k_grid[i, j] = k if k is not None else np.nan
            tag = f"k={k:+.4f}" if k is not None else "insuff."
            flip = "FLIP" if k is not None and k < 0 else ""
            print(f"  eta={eta:.2f}  M={M:2d}  {tag:>14s}  {flip}")

    # ── Plot heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))

    # Diverging colour map centred on k=0
    vmax = np.nanmax(np.abs(k_grid))
    vmax = max(vmax, 0.05)   # floor so flat maps still have contrast
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(k_grid, aspect='auto', origin='lower',
                   cmap='RdBu', norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("k  (curvature at n+ = 0.5)", fontsize=11)

    # Annotate each cell
    for i in range(len(ETA_LIST)):
        for j in range(len(M_LIST)):
            val = k_grid[i, j]
            if np.isnan(val):
                txt, col = "—", "gray"
            else:
                txt = f"{val:+.3f}"
                col = "white" if abs(val) > 0.4 * vmax else "black"
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=col)

    # Overlay: thick border around cells where k < 0 (FLIP)
    for i in range(len(ETA_LIST)):
        for j in range(len(M_LIST)):
            val = k_grid[i, j]
            if val is not None and not np.isnan(val) and val < 0:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor='lime',
                                     facecolor='none', zorder=10)
                ax.add_patch(rect)

    ax.set_xticks(range(len(M_LIST)))
    ax.set_xticklabels([str(m) for m in M_LIST])
    ax.set_yticks(range(len(ETA_LIST)))
    ax.set_yticklabels([f"{e:.2f}" for e in ETA_LIST])
    ax.set_xlabel("Memory depth  M", fontsize=12)
    ax.set_ylabel("Observation noise  η", fontsize=12)
    ax.set_title(
        f"Market-Only Population — Signal Flip Heatmap\n"
        f"Behavior: {beh.upper()}   |   Blue = single well (k > 0, no flip)   |   "
        f"Red = double well (k < 0, FLIP)   |   Green border = FLIP",
        fontsize=11, pad=12)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"heatmap_market_only_{beh}.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → saved heatmap_market_only_{beh}.png\n")

print("✅  All heatmaps generated.")
