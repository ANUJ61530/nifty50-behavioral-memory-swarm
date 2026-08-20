"""
T_1/2 Half-Life Relaxation Experiment
======================================
Matches Li et al. Fig 4(b)-(c): starting from unanimous -1 consensus,
measure how long until n+ reaches 0.5 for each personality / M / b.
Run on real Nifty 50 TRI data.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATE_COL     = "timestamp"
CLOSE_COL    = "close"

ETA          = 0.3       # detection error (paper's measured value)
N_SEEDS      = 150       # Monte Carlo population size
P_PEER       = 1.0       # peer-interaction probability
TREND_WINDOW = 200       # curmudgeon MA window
RANDOM_SEED  = 0

M_LIST       = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
B_OPP_LIST   = [1.1, 1.3, 1.5, 1.8, 2.0]     # opportunist bias sweep
B_TRAD_LIST  = [0.9, 0.8, 0.7, 0.6, 0.5]      # traditionalist bias sweep

THEORETICAL_MC = np.pi / (2 * (1 - 2 * ETA) ** 2)
print(f"Theoretical M_c (eta={ETA}): {THEORETICAL_MC:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_returns():
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["ret"] = df[CLOSE_COL].pct_change()
    df["q"] = np.sign(df["ret"])
    df.loc[df["q"] == 0, "q"] = 1.0
    # Curmudgeon trend signal
    df["ma200"] = df[CLOSE_COL].rolling(window=TREND_WINDOW).mean()
    df["d_ma200"] = df["ma200"].diff()
    df["C"] = np.sign(df["d_ma200"].shift(1))
    df["C"] = df["C"].fillna(-1.0)   # default -1 during warm-up (matches unanimous start)
    df.loc[df["C"] == 0, "C"] = 1.0
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    df["q_lagged"] = df["q"].shift(1)
    return df

# ═══════════════════════════════════════════════════════════════════════
# WEIGHT VECTOR
# ═══════════════════════════════════════════════════════════════════════
def make_weights(personality, M, b=1.5):
    if personality == "pushover":
        w = np.ones(M)
    elif personality == "opportunist":
        w = b ** np.arange(M)          # newest slot = index M-1 gets largest weight
    elif personality == "traditionalist":
        if b < 1.0:
            w = b ** np.arange(M)      # oldest slot = index 0 gets largest weight
        else:
            w = b ** (-np.arange(M))
    elif personality == "contrarian":
        w = np.ones(M)                 # same as pushover; sign flip applied at decision
    elif personality == "curmudgeon":
        w = np.ones(M)                 # placeholder (curmudgeons ignore queue)
    else:
        raise ValueError(f"Unknown personality: {personality}")
    s = np.sum(np.abs(w))
    return w / s if s > 0 else w

# ═══════════════════════════════════════════════════════════════════════
# RELAXATION SIMULATOR — starts from unanimous -1
# ═══════════════════════════════════════════════════════════════════════
def simulate_relaxation_from_unanimous(
    q_market, C_market, M, eta, p_peer, n_seeds,
    personality="pushover", b=1.5,
    curmudgeon_frac=0.0, seed_base=RANDOM_SEED
):
    """
    Returns
    -------
    n_plus_series : np.ndarray   full n+(t) trajectory
    t_half        : int or None  first t where n+(t) >= 0.5
    """
    n_days = len(q_market)
    rng = np.random.RandomState(seed_base)

    # ── assign types ──────────────────────────────────────────────────
    n_curm = int(n_seeds * curmudgeon_frac)
    n_active = n_seeds - n_curm
    is_curmudgeon = np.zeros(n_seeds, dtype=bool)
    is_curmudgeon[:n_curm] = True
    is_contrarian = np.zeros(n_seeds, dtype=bool)
    if personality == "contrarian":
        is_contrarian[n_curm:] = True

    # ── weight vector (same for all active agents) ────────────────────
    weights = make_weights(personality, M, b)

    # ── unanimous -1 start ────────────────────────────────────────────
    buffer = -np.ones((n_seeds, M))
    n_plus_series = np.full(n_days, np.nan)
    prev_n_plus = 0.0   # by construction: all start at -1
    t_half = None

    for t in range(n_days):
        if np.isnan(q_market[t]):
            continue
        pos = t % M

        # ── observation ───────────────────────────────────────────────
        use_peer = rng.rand(n_seeds) < p_peer
        peer_bit = np.where(rng.rand(n_seeds) < prev_n_plus, 1.0, -1.0)
        market_bit = np.full(n_seeds, q_market[t])
        raw_bit = np.where(use_peer, peer_bit, market_bit)
        flip = rng.rand(n_seeds) < eta
        new_bit = np.where(flip, -raw_bit, raw_bit)

        # curmudgeons don't update their buffer (it stays -1 forever)
        buffer[~is_curmudgeon, pos] = new_bit[~is_curmudgeon]

        # ── decision ──────────────────────────────────────────────────
        if t >= M - 1:
            chron_buffer = np.roll(buffer, -pos - 1, axis=1)
            sigma = np.dot(chron_buffer, weights)     # (n_seeds,)
            opinions = np.sign(sigma)
            opinions[opinions == 0] = 1.0

            # contrarians negate their own vote
            opinions[is_contrarian] = -opinions[is_contrarian]

            # curmudgeons report fixed bias
            opinions[is_curmudgeon] = C_market[t]

            n_plus = np.mean(opinions == 1.0)
            n_plus_series[t] = n_plus
            prev_n_plus = n_plus

            if t_half is None and n_plus >= 0.5:
                t_half = t

    return n_plus_series, t_half

# ═══════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════
def run_sanity_checks():
    q_syn = np.ones(100)

    # 1) 100% curmudgeon, fixed bias = -1 every day → n+ stays 0, T_1/2 = None
    C_neg = -np.ones(100)
    _, th = simulate_relaxation_from_unanimous(
        q_syn, C_neg, M=3, eta=0, p_peer=1.0, n_seeds=50,
        personality="pushover", curmudgeon_frac=1.0)
    assert th is None, f"Sanity 1 FAILED: curmudgeon -1 should never reach 0.5 (got {th})"

    # 2) 100% curmudgeon, fixed bias = +1 → T_1/2 = M-1 (first valid timestep)
    C_pos = np.ones(100)
    _, th = simulate_relaxation_from_unanimous(
        q_syn, C_pos, M=3, eta=0, p_peer=1.0, n_seeds=50,
        personality="pushover", curmudgeon_frac=1.0)
    assert th == 2, f"Sanity 2 FAILED: curmudgeon +1 should reach 0.5 at t=M-1=2 (got {th})"

    # 3) 100% contrarian, p_peer=1.0, unanimous -1 start:
    #    prev_n_plus=0 → peers always -1 → buffer stays -1 → majority=-1
    #    contrarian negates to +1 → T_1/2 = M-1 (first valid timestep)
    C_dummy = np.zeros(100)
    _, th = simulate_relaxation_from_unanimous(
        q_syn, C_dummy, M=3, eta=0, p_peer=1.0, n_seeds=50,
        personality="contrarian")
    assert th == 2, f"Sanity 3 FAILED: contrarian should flip immediately (got {th})"

    print("✅ All sanity checks passed.")

run_sanity_checks()

# ═══════════════════════════════════════════════════════════════════════
# LOAD REAL DATA
# ═══════════════════════════════════════════════════════════════════════
df = load_returns()
q_market = df["q_lagged"].values
C_market = df["C"].values
n_days = len(q_market)
print(f"Loaded {n_days} trading days of Nifty 50 data.\n")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 1 — N+ growth curves: pushover vs opportunist vs traditionalist
#           at a single representative M, showing different relaxation speeds
# ═══════════════════════════════════════════════════════════════════════
print("── Graph 1: N+ growth curves ──")
M_demo = 7
fig, ax = plt.subplots(figsize=(10, 6))
styles = {
    ("pushover",       1.0,  "steelblue",  "-",  "Pushover (b=1.0)"),
    ("opportunist",    1.5,  "darkorange", "-",  "Opportunist (b=1.5)"),
    ("traditionalist", 0.7,  "seagreen",   "-",  "Traditionalist (b=0.7)"),
    ("contrarian",     1.0,  "crimson",    "--", "Contrarian"),
}
for pers, b, col, ls, lbl in sorted(styles, key=lambda x: x[0]):
    nps, th = simulate_relaxation_from_unanimous(
        q_market, C_market, M_demo, ETA, P_PEER, N_SEEDS,
        personality=pers, b=b)
    valid = ~np.isnan(nps)
    ax.plot(np.where(valid)[0], nps[valid], color=col, ls=ls, lw=1.8, label=lbl)
    if th is not None:
        ax.axvline(th, color=col, ls=":", lw=0.9, alpha=0.6)
        ax.text(th + 5, 0.05, f"T½={th}", fontsize=8, color=col)

ax.axhline(0.5, color="black", lw=0.8, ls="--", label="n+ = 0.5 (target)")
ax.set_xlabel("Trading day t")
ax.set_ylabel("n+(t)")
ax.set_title(f"Relaxation from unanimous −1  |  M={M_demo}, η={ETA}, N={N_SEEDS}, p_peer={P_PEER}")
ax.legend(fontsize=9)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_growth_curves.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_growth_curves.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 2 — T_1/2 vs M for pushover, opportunist, traditionalist, contrarian
#           (paper's Fig 4b analog)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 2: T½ vs M (all behaviors at default b) ──")
behaviors_g2 = [
    ("pushover",       1.0,  "steelblue",  "o"),
    ("opportunist",    1.5,  "darkorange", "s"),
    ("traditionalist", 0.7,  "seagreen",   "^"),
    ("contrarian",     1.0,  "crimson",    "D"),
]

fig, ax = plt.subplots(figsize=(10, 6))
for pers, b, col, mk in behaviors_g2:
    ths = []
    for M in M_LIST:
        _, th = simulate_relaxation_from_unanimous(
            q_market, C_market, M, ETA, P_PEER, N_SEEDS,
            personality=pers, b=b)
        ths.append(th)
        tag = str(th) if th is not None else "DNF"
        print(f"  {pers:15s} b={b:.1f}  M={M:2d}  T½={tag}")

    # plot — use n_days as a cap for DNF (didn't finish) points
    ths_plot = [t if t is not None else n_days for t in ths]
    dnf_mask = [t is None for t in ths]

    ax.plot(M_LIST, ths_plot, marker=mk, color=col, lw=1.8,
            label=f"{pers} (b={b})")
    # mark DNF points with an open marker
    for i, dnf in enumerate(dnf_mask):
        if dnf:
            ax.plot(M_LIST[i], n_days, marker=mk, color=col,
                    markersize=10, markerfacecolor="white", markeredgewidth=2)

ax.axvline(THEORETICAL_MC, color="red", ls="--", lw=1.2,
           label=f"theoretical M_c = {THEORETICAL_MC:.1f}")
ax.set_xlabel("Memory depth M")
ax.set_ylabel("T½ (days to reach n+ = 0.5)")
ax.set_title(f"Half-Life vs Memory Depth  |  η={ETA}, N={N_SEEDS}, p_peer={P_PEER}\n"
             f"(open markers = did not converge within {n_days} days)")
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_vs_M.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_vs_M.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 3a — T_1/2 heatmap: OPPORTUNIST (b × M)  [paper's Fig 4c analog]
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 3a: Opportunist T½ heatmap (b × M) ──")
thalf_opp = np.full((len(B_OPP_LIST), len(M_LIST)), np.nan)
for i, b in enumerate(B_OPP_LIST):
    for j, M in enumerate(M_LIST):
        _, th = simulate_relaxation_from_unanimous(
            q_market, C_market, M, ETA, P_PEER, N_SEEDS,
            personality="opportunist", b=b)
        thalf_opp[i, j] = th if th is not None else n_days
        tag = str(th) if th is not None else "DNF"
        print(f"  opportunist  b={b:.1f}  M={M:2d}  T½={tag}")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(thalf_opp, aspect="auto", origin="lower",
               cmap="YlOrRd", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("T½ (days)", fontsize=11)
for i in range(len(B_OPP_LIST)):
    for j in range(len(M_LIST)):
        v = thalf_opp[i, j]
        txt = "DNF" if v >= n_days else f"{int(v)}"
        c = "white" if v > thalf_opp[~np.isnan(thalf_opp)].mean() else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=c)
ax.set_xticks(range(len(M_LIST)))
ax.set_xticklabels([str(m) for m in M_LIST])
ax.set_yticks(range(len(B_OPP_LIST)))
ax.set_yticklabels([f"{b:.1f}" for b in B_OPP_LIST])
ax.set_xlabel("Memory depth M", fontsize=12)
ax.set_ylabel("Bias parameter b", fontsize=12)
ax.set_title(f"Opportunist T½ Heatmap  |  η={ETA}, N={N_SEEDS}, p_peer={P_PEER}", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_heatmap_opportunist.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_heatmap_opportunist.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 3b — T_1/2 heatmap: TRADITIONALIST (b × M)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 3b: Traditionalist T½ heatmap (b × M) ──")
thalf_trad = np.full((len(B_TRAD_LIST), len(M_LIST)), np.nan)
for i, b in enumerate(B_TRAD_LIST):
    for j, M in enumerate(M_LIST):
        _, th = simulate_relaxation_from_unanimous(
            q_market, C_market, M, ETA, P_PEER, N_SEEDS,
            personality="traditionalist", b=b)
        thalf_trad[i, j] = th if th is not None else n_days
        tag = str(th) if th is not None else "DNF"
        print(f"  traditionalist  b={b:.1f}  M={M:2d}  T½={tag}")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(thalf_trad, aspect="auto", origin="lower",
               cmap="YlOrRd", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("T½ (days)", fontsize=11)
for i in range(len(B_TRAD_LIST)):
    for j in range(len(M_LIST)):
        v = thalf_trad[i, j]
        txt = "DNF" if v >= n_days else f"{int(v)}"
        c = "white" if v > thalf_trad[~np.isnan(thalf_trad)].mean() else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=c)
ax.set_xticks(range(len(M_LIST)))
ax.set_xticklabels([str(m) for m in M_LIST])
ax.set_yticks(range(len(B_TRAD_LIST)))
ax.set_yticklabels([f"{b:.1f}" for b in B_TRAD_LIST])
ax.set_xlabel("Memory depth M", fontsize=12)
ax.set_ylabel("Bias parameter b", fontsize=12)
ax.set_title(f"Traditionalist T½ Heatmap  |  η={ETA}, N={N_SEEDS}, p_peer={P_PEER}", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_heatmap_traditionalist.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_heatmap_traditionalist.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 4 — T_1/2 vs M: overlay opportunist at different b values
#           (line plot version of the heatmap, easier to read trends)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 4: T½ vs M at multiple b (opportunist & traditionalist) ──")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Opportunist
ax = axes[0]
cmap_o = plt.cm.Oranges(np.linspace(0.3, 0.9, len(B_OPP_LIST)))
for i, b in enumerate(B_OPP_LIST):
    ths = thalf_opp[i, :]
    ths_plot = [t if t < n_days else n_days for t in ths]
    ax.plot(M_LIST, ths_plot, marker="o", color=cmap_o[i], lw=1.5, label=f"b={b:.1f}")
ax.axvline(THEORETICAL_MC, color="red", ls="--", lw=1)
ax.set_xlabel("Memory depth M")
ax.set_ylabel("T½ (days)")
ax.set_title("Opportunist: T½ vs M at different b")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(2))

# Traditionalist
ax = axes[1]
cmap_t = plt.cm.Greens(np.linspace(0.3, 0.9, len(B_TRAD_LIST)))
for i, b in enumerate(B_TRAD_LIST):
    ths = thalf_trad[i, :]
    ths_plot = [t if t < n_days else n_days for t in ths]
    ax.plot(M_LIST, ths_plot, marker="s", color=cmap_t[i], lw=1.5, label=f"b={b:.1f}")
ax.axvline(THEORETICAL_MC, color="red", ls="--", lw=1)
ax.set_xlabel("Memory depth M")
ax.set_ylabel("T½ (days)")
ax.set_title("Traditionalist: T½ vs M at different b")
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(2))

plt.suptitle(f"Half-Life Curves  |  η={ETA}, N={N_SEEDS}", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_curves_b_sweep.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → saved thalf_curves_b_sweep.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 5 — Growth curves at multiple M for pushover (critical slowing down)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 5: Pushover n+ growth at multiple M (critical slowing down) ──")
fig, ax = plt.subplots(figsize=(11, 6))
cmap_m = plt.cm.viridis(np.linspace(0, 1, len(M_LIST)))
for idx, M in enumerate(M_LIST):
    nps, th = simulate_relaxation_from_unanimous(
        q_market, C_market, M, ETA, P_PEER, N_SEEDS,
        personality="pushover")
    valid = ~np.isnan(nps)
    ts = np.where(valid)[0]
    # clip to first 500 days for readability
    mask = ts < 500
    ax.plot(ts[mask], nps[valid][mask], color=cmap_m[idx], lw=1.3,
            label=f"M={M}" + (f" T½={th}" if th is not None and th < 500 else ""))

ax.axhline(0.5, color="black", lw=0.8, ls="--")
ax.set_xlabel("Trading day t")
ax.set_ylabel("n+(t)")
ax.set_title(f"Pushover Relaxation — Critical Slowing Down Near M_c\n"
             f"η={ETA}, N={N_SEEDS}, p_peer={P_PEER}")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_pushover_critical_slowing.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_pushover_critical_slowing.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 6 — Combined heatmap: Pushover T½ heatmap (b=1) + 12.5% curmudgeon
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 6: Curmudgeon nucleation effect on T½ ──")
curm_fracs = [0.0, 0.05, 0.10, 0.125, 0.20, 0.30]
thalf_curm = np.full((len(curm_fracs), len(M_LIST)), np.nan)
for i, cf in enumerate(curm_fracs):
    for j, M in enumerate(M_LIST):
        _, th = simulate_relaxation_from_unanimous(
            q_market, C_market, M, ETA, P_PEER, N_SEEDS,
            personality="pushover", curmudgeon_frac=cf)
        thalf_curm[i, j] = th if th is not None else n_days
        tag = str(th) if th is not None else "DNF"
        print(f"  pushover+curm={cf:.0%}  M={M:2d}  T½={tag}")

fig, ax = plt.subplots(figsize=(12, 5.5))
im = ax.imshow(thalf_curm, aspect="auto", origin="lower",
               cmap="YlOrRd", interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("T½ (days)", fontsize=11)
for i in range(len(curm_fracs)):
    for j in range(len(M_LIST)):
        v = thalf_curm[i, j]
        txt = "DNF" if v >= n_days else f"{int(v)}"
        c = "white" if v > thalf_curm[~np.isnan(thalf_curm)].mean() else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=c)
ax.set_xticks(range(len(M_LIST)))
ax.set_xticklabels([str(m) for m in M_LIST])
ax.set_yticks(range(len(curm_fracs)))
ax.set_yticklabels([f"{cf:.0%}" for cf in curm_fracs])
ax.set_xlabel("Memory depth M", fontsize=12)
ax.set_ylabel("Curmudgeon fraction", fontsize=12)
ax.set_title(f"Curmudgeon Nucleation: Effect on Pushover T½  |  η={ETA}", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "thalf_heatmap_curmudgeon_nucleation.png"), dpi=150)
plt.close(fig)
print("  → saved thalf_heatmap_curmudgeon_nucleation.png")

print("\n✅ T½ relaxation experiment complete. All plots saved to:", OUTPUT_DIR)
