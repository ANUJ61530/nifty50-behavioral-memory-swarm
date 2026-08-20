import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config constants
ETA = 0.3
N_SEEDS = 150
M_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
P_PEER = 1.0
N_BINS = 20
RANDOM_SEED = 0

THEORETICAL_MC = np.pi / (2 * (1 - 2 * ETA) ** 2)
print(f"Theoretical critical memory depth for eta={ETA}: M_c = {THEORETICAL_MC:.2f}")

# ── Data loading ──────────────────────────────────────────────────────
def load_returns(path=DATA_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["q"] = np.sign(df["ret"])
    df.loc[df["q"] == 0, "q"] = 1.0

    # Pre-calculate Curmudgeon signal
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

# ── Weight vectors ────────────────────────────────────────────────────
def get_agent_weights(personality, M, b_opp=1.5, b_trad=0.7):
    if personality == 'pushover':
        w = np.ones(M)
    elif personality == 'opportunist':
        w = b_opp ** np.arange(M)
    elif personality == 'traditionalist':
        if b_trad < 1.0:
            w = b_trad ** np.arange(M)
        else:
            w = b_trad ** (-np.arange(M))
    elif personality == 'contrarian':
        w = -np.ones(M)
    else:  # curmudgeon fallback
        w = np.ones(M)
    abs_sum = np.sum(np.abs(w))
    if abs_sum > 0:
        w = w / abs_sum
    return w

# ── Mixed-behaviour simulation (supports p_peer=0 for market-only) ───
def simulate_population_n_plus_mixed(
    q_market, C_market, M, eta, p_peer, n_seeds,
    type_proportions, b_opp=1.5, b_trad=0.7, seed_base=RANDOM_SEED
):
    n_days = len(q_market)
    rng = np.random.RandomState(seed_base)

    # 1. Assign behaviour types
    types = ['pushover', 'opportunist', 'traditionalist', 'contrarian', 'curmudgeon']
    num_agents_per_type = {}
    remaining = n_seeds
    type_list = []
    for type_name in types[:-1]:
        num = int(n_seeds * type_proportions.get(type_name, 0.0))
        num_agents_per_type[type_name] = num
        remaining -= num
    num_agents_per_type['curmudgeon'] = max(0, remaining)
    for type_name, num in num_agents_per_type.items():
        type_list.extend([type_name] * num)
    type_array = np.array(type_list)[:n_seeds]
    rng.shuffle(type_array)
    is_curmudgeon = (type_array == 'curmudgeon')

    # 2. Weight matrix
    weights = np.zeros((n_seeds, M))
    for i in range(n_seeds):
        weights[i, :] = get_agent_weights(type_array[i], M, b_opp, b_trad)

    # 3. Day-by-day loop
    buffer = np.zeros((n_seeds, M))
    n_plus_series = np.full(n_days, np.nan)
    for t in range(n_days):
        if np.isnan(q_market[t]):
            continue
        pos = t % M
        if t < M:
            use_peer = np.zeros(n_seeds, dtype=bool)
            peer_bit = np.zeros(n_seeds)
        else:
            prev_n_plus = n_plus_series[t - 1]
            use_peer = rng.rand(n_seeds) < p_peer
            peer_bit = np.where(rng.rand(n_seeds) < prev_n_plus, 1.0, -1.0)
        market_bit = np.full(n_seeds, q_market[t])
        raw_bit = np.where(use_peer, peer_bit, market_bit)
        flip = rng.rand(n_seeds) < eta
        new_bit = np.where(flip, -raw_bit, raw_bit)
        buffer[:, pos] = new_bit
        if t >= M - 1:
            chron_buffer = np.roll(buffer, -pos - 1, axis=1)
            sigma = np.sum(chron_buffer * weights, axis=1)
            opinions = np.sign(sigma)
            opinions[opinions == 0] = 1.0
            opinions[is_curmudgeon] = C_market[t]
            n_plus_series[t] = np.mean(opinions == 1.0)
    return n_plus_series

# ── Drift / potential fitting ─────────────────────────────────────────
def fit_drift_and_potential(n_plus, n_bins=N_BINS):
    x = n_plus[:-1] - 0.5
    dx = n_plus[1:] - n_plus[:-1]
    valid = ~np.isnan(x) & ~np.isnan(dx)
    x, dx = x[valid], dx[valid]
    if len(x) < 30:
        return None
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx = np.digitize(x, bin_edges) - 1
    bin_centers, bin_drift = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() >= 5:
            bin_centers.append(x[mask].mean())
            bin_drift.append(dx[mask].mean())
    bin_centers = np.array(bin_centers)
    bin_drift = np.array(bin_drift)
    if len(bin_centers) < 4:
        return None
    design = np.column_stack([bin_centers, bin_centers ** 3])
    coeffs, *_ = np.linalg.lstsq(design, bin_drift, rcond=None)
    a1, a3 = coeffs
    k, c = -a1, -a3
    return {"bin_centers": bin_centers, "bin_drift": bin_drift,
            "k": k, "c": c, "x_raw": x, "dx_raw": dx}

def potential_V(x, k, c):
    return 0.5 * k * x ** 2 + 0.25 * c * x ** 4

# ── Sanity checks ─────────────────────────────────────────────────────
def run_sanity_checks():
    q_test = np.ones(10)
    C_test = np.ones(10)
    n_plus = simulate_population_n_plus_mixed(
        q_test, C_test, M=3, eta=0.0, p_peer=0.0, n_seeds=50,
        type_proportions={'pushover': 1.0})
    assert np.all(n_plus[~np.isnan(n_plus)] == 1.0), "Pushover sanity failed"
    n_plus_c = simulate_population_n_plus_mixed(
        q_test, C_test, M=3, eta=0.0, p_peer=0.0, n_seeds=50,
        type_proportions={'contrarian': 1.0})
    assert np.all(n_plus_c[~np.isnan(n_plus_c)] == 0.0), "Contrarian sanity failed"
    print("All sanity checks passed.")

run_sanity_checks()

# ── Load Nifty data ───────────────────────────────────────────────────
df = load_returns()
df = build_lagged_q(df)
q_market = df["q_lagged"].values
C_market = df["C"].values

# ── Cases ─────────────────────────────────────────────────────────────
cases = {
    "Case 1a - Pure Pushover": {"pushover": 1.0},
    "Case 1b - Pure Opportunist": {"opportunist": 1.0},
    "Case 1c - Pure Traditionalist": {"traditionalist": 1.0},
    "Case 1d - Pure Contrarian": {"contrarian": 1.0},
    "Case 2 - Equal Mix (20pct each)": {
        "pushover": 0.2, "opportunist": 0.2, "traditionalist": 0.2,
        "contrarian": 0.2, "curmudgeon": 0.2},
    "Case 3 - 80pct Pushover 20pct Contrarian": {"pushover": 0.8, "contrarian": 0.2},
    "Case 4 - 90pct Pushover 10pct Curmudgeon": {"pushover": 0.9, "curmudgeon": 0.1},
    "Case 5 - 50pct Opportunist 50pct Traditionalist": {"opportunist": 0.5, "traditionalist": 0.5},
}

# ── Sweep both crowds for every case ─────────────────────────────────
for case_name, props in cases.items():
    safe_name = case_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
    print(f"\n{'='*72}")
    print(f"  {case_name}")
    print(f"{'='*72}")

    sweep_market = {}   # market-only  (p_peer = 0)
    sweep_peer   = {}   # peer-interacting (p_peer = P_PEER)

    for M in M_LIST:
        # ── market-only crowd ──
        n_plus_m = simulate_population_n_plus_mixed(
            q_market, C_market, M, ETA, 0.0, N_SEEDS, props)
        fit_m = fit_drift_and_potential(n_plus_m)
        sweep_market[M] = {"n_plus": n_plus_m, "fit": fit_m}

        # ── peer-interacting crowd ──
        n_plus_p = simulate_population_n_plus_mixed(
            q_market, C_market, M, ETA, P_PEER, N_SEEDS, props)
        fit_p = fit_drift_and_potential(n_plus_p)
        sweep_peer[M] = {"n_plus": n_plus_p, "fit": fit_p}

        tag_m = (f"k={fit_m['k']:+.4f} ({'double' if fit_m['k']<0 else 'single'} well)"
                 if fit_m else "insufficient data")
        tag_p = (f"k={fit_p['k']:+.4f} ({'double' if fit_p['k']<0 else 'single'} well)"
                 if fit_p else "insufficient data")
        print(f"  M={M:2d}  market-only: {tag_m}   |   peer(p={P_PEER}): {tag_p}")

    # ── Graph 1: k vs M  (both crowds) ────────────────────────────────
    Ms_m  = [M for M in M_LIST if sweep_market[M]["fit"] is not None]
    ks_m  = [sweep_market[M]["fit"]["k"] for M in Ms_m]
    Ms_p  = [M for M in M_LIST if sweep_peer[M]["fit"]   is not None]
    ks_p  = [sweep_peer[M]["fit"]["k"]   for M in Ms_p]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(Ms_m, ks_m, marker="o", color="steelblue",
            label="market-only population (no peer interaction)")
    ax.plot(Ms_p, ks_p, marker="s", color="darkorange",
            label=f"peer-interacting population (p_peer={P_PEER})")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(THEORETICAL_MC, color="red", linestyle="--",
               label=f"theoretical M_c = {THEORETICAL_MC:.1f} (eta={ETA})")
    ax.set_xlabel("Memory depth M")
    ax.set_ylabel("k (curvature at n+ = 0.5)")
    ax.set_title(f"Curvature vs Memory Depth — {case_name}")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(min(M_LIST), max(M_LIST))
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_k_vs_M.png"), dpi=150)
    plt.close(fig)

    # ── Graph 2: Drift functions — small M & large M, both crowds ─────
    M_small, M_large = min(M_LIST), max(M_LIST)
    for M_drift in [M_small, M_large]:
        fit_m = sweep_market[M_drift]["fit"]
        fit_p = sweep_peer[M_drift]["fit"]
        if fit_m is None and fit_p is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: market-only
        ax = axes[0]
        if fit_m is not None:
            ax.scatter(fit_m["x_raw"]+0.5, fit_m["dx_raw"], alpha=.12, s=8, color="gray")
            ax.scatter(fit_m["bin_centers"]+0.5, fit_m["bin_drift"],
                       color="steelblue", s=50, zorder=5, label="binned mean drift")
            xf = np.linspace(-0.5, 0.5, 200)
            ax.plot(xf+0.5, -(fit_m["k"]*xf + fit_m["c"]*xf**3),
                    color="navy", lw=2, label="fitted drift")
            ax.set_title(f"Market-only, M={M_drift} (k={fit_m['k']:+.3f})")
        else:
            ax.set_title(f"Market-only, M={M_drift}: insufficient data")
        ax.axhline(0, color="black", lw=.8); ax.axvline(0.5, color="black", lw=.8, ls=":")
        ax.set_xlabel("n+(t)"); ax.set_ylabel("drift")
        ax.legend(fontsize=8)

        # Right: peer
        ax = axes[1]
        if fit_p is not None:
            ax.scatter(fit_p["x_raw"]+0.5, fit_p["dx_raw"], alpha=.12, s=8, color="gray")
            ax.scatter(fit_p["bin_centers"]+0.5, fit_p["bin_drift"],
                       color="darkorange", s=50, zorder=5, label="binned mean drift")
            xf = np.linspace(-0.5, 0.5, 200)
            ax.plot(xf+0.5, -(fit_p["k"]*xf + fit_p["c"]*xf**3),
                    color="darkred", lw=2, label="fitted drift")
            ax.set_title(f"Peer-interacting, M={M_drift} (k={fit_p['k']:+.3f})")
        else:
            ax.set_title(f"Peer-interacting, M={M_drift}: insufficient data")
        ax.axhline(0, color="black", lw=.8); ax.axvline(0.5, color="black", lw=.8, ls=":")
        ax.set_xlabel("n+(t)"); ax.set_ylabel("drift")
        ax.legend(fontsize=8)

        plt.suptitle(f"Drift Functions — {case_name}, M={M_drift}", fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_drift_M{M_drift}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Graph 3: Potential landscape — both crowds side-by-side ────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xf = np.linspace(-0.5, 0.5, 200)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(M_LIST)))

    for ax_idx, (sweep, crowd_label) in enumerate(
            [(sweep_market, "Market-only"), (sweep_peer, f"Peer-interacting (p={P_PEER})")]):
        ax = axes[ax_idx]
        for M, col in zip(M_LIST, cmap):
            fit = sweep[M]["fit"]
            if fit is None:
                continue
            V = potential_V(xf, fit["k"], fit["c"])
            V = V - V.min()
            ax.plot(xf + 0.5, V, label=f"M={M}", color=col)
        ax.set_xlabel("n+")
        ax.set_ylabel("V(n+)  (normalised)")
        ax.set_title(f"{crowd_label}")
        ax.legend(fontsize=7, ncol=2)

    plt.suptitle(f"Potential Landscape — {case_name}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{safe_name}_potential.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

print("\n✅ All cases completed. Plots saved to:", OUTPUT_DIR)
