import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Returns
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df["q"] = np.sign(df["ret"])
df.loc[df["q"] == 0, "q"] = 1.0

# 200-day trend signal
df["ma200"] = df["close"].rolling(window=200).mean()
df["d_ma200"] = df["ma200"].diff()
df["C"] = np.sign(df["d_ma200"].shift(1))
df["C"] = df["C"].fillna(-1.0)
df.loc[df["C"] == 0, "C"] = 1.0

df = df.dropna(subset=["ret"]).reset_index(drop=True)
df["q_lagged"] = df["q"].shift(1)

q_market = df["q_lagged"].values
C_market = df["C"].values
n_days = len(q_market)

# Config
ETA = 0.30
N_SEEDS = 150
P_PEER = 1.0
NUM_MC_RUNS = 10

# Memory depths to compare (matching Li et al. multi-curve layout)
M_TARGETS = [5, 7, 9, 11, 15, 21]

# Continuous b grid from Traditionalist (b < 1) through Pushover (b = 1) to Opportunist (b > 1)
B_GRID = np.concatenate([
    np.linspace(0.40, 0.95, 12),
    np.array([1.0]),
    np.linspace(1.05, 2.20, 15)
])
B_GRID = np.unique(np.sort(B_GRID))

def make_continuous_weights(M, b):
    # k = 0 (oldest) to k = M-1 (newest)
    w = (b ** np.arange(M)).astype(float)
    s = np.sum(np.abs(w))
    return w / s if s > 0 else w

def simulate_relaxation(q_market, C_market, M, eta, p_peer, n_seeds, b, seed_base=0):
    rng = np.random.RandomState(seed_base)
    weights = make_continuous_weights(M, b)
    
    # Start with unanimous -1 (panic state)
    buffer = -np.ones((n_seeds, M))
    prev_n_plus = 0.0
    t_half = None
    
    for t in range(n_days):
        if np.isnan(q_market[t]):
            continue
        pos = t % M
        
        # Peer vs Market observation
        use_peer = rng.rand(n_seeds) < p_peer
        peer_bit = np.where(rng.rand(n_seeds) < prev_n_plus, 1.0, -1.0)
        market_bit = np.full(n_seeds, q_market[t])
        raw_bit = np.where(use_peer, peer_bit, market_bit)
        
        # Cognitive noise channel BSC(eta)
        flip = rng.rand(n_seeds) < eta
        new_bit = np.where(flip, -raw_bit, raw_bit)
        buffer[:, pos] = new_bit
        
        # Decision at t >= M-1
        if t >= M - 1:
            chron_buffer = np.roll(buffer, -pos - 1, axis=1)
            sigma = np.dot(chron_buffer, weights)
            opinions = np.sign(sigma)
            opinions[opinions == 0] = 1.0
            
            n_plus = np.mean(opinions == 1.0)
            prev_n_plus = n_plus
            
            if t_half is None and n_plus >= 0.50:
                t_half = t
                break  # Relaxation half-life reached!
                
    # If not reached within sample, return n_days (or None)
    return t_half if t_half is not None else n_days

print("Running relaxation sweep across continuous b grid and memory depths M...")

results = {M: [] for M in M_TARGETS}

for M in M_TARGETS:
    print(f"  Evaluating Memory Depth M = {M:2d}...")
    for b in B_GRID:
        thalf_runs = []
        for seed in range(NUM_MC_RUNS):
            th = simulate_relaxation(q_market, C_market, M, ETA, P_PEER, N_SEEDS, b, seed_base=seed*100 + int(b*1000) + M)
            thalf_runs.append(th)
        avg_thalf = np.median(thalf_runs)
        results[M].append(avg_thalf)

print("Simulation complete. Generating plot...")

# ==============================================================================
# PLOT: T_1/2 vs. Bias Parameter b (Matching Li et al. Figure 4(c) Transpose Layout)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 7.0), dpi=250)

# Palette for curves
curve_colors = {
    5: '#1b9e77',    # Green
    7: '#386cb0',    # Blue
    9: '#7570b3',    # Purple
    11: '#e7298a',   # Magenta / Pink
    15: '#d95f02',   # Orange
    21: '#e41a1c'    # Red
}

for M in M_TARGETS:
    y_vals = np.array(results[M])
    # Cap DNF for visual clarity if needed
    ax.plot(B_GRID, y_vals, marker='o', markersize=5.5, linewidth=2.4, 
            color=curve_colors[M], label=f'$M = {M}$ days')

# Vertical line at b = 1.0 (Pushover / neutral memory weighting)
ax.axvline(1.0, color='black', linestyle='--', linewidth=1.8, alpha=0.85, label='Pushover Baseline ($b = 1.0$)')

# Shaded background zones for Traditionalist (b < 1) vs Opportunist (b > 1)
ax.axvspan(0.35, 1.0, color='#fee8c8', alpha=0.35)
ax.axvspan(1.0, 2.25, color='#e0f3f8', alpha=0.35)

# Zone annotations
ax.text(0.65, 1800, r"$\mathbf{\longleftarrow\ Traditionalist\ (b < 1)}$" "\n(Distance-Weighted Memory\nExtended Hysteresis / Lock-In)", 
        fontsize=10.5, fontweight='bold', color='#b30000', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#e34a33', alpha=0.90))

ax.text(1.65, 1800, r"$\mathbf{Opportunist\ (b > 1)\ \longrightarrow}$" "\n(Recency-Weighted Memory\nAccelerated Panic Recovery)", 
        fontsize=10.5, fontweight='bold', color='#08519c', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2171b5', alpha=0.90))

# Axis styling & annotations
ax.set_xlim(0.35, 2.25)
ax.set_ylim(0, 2600)
ax.set_xlabel(r"$\longleftarrow\ \mathbf{Traditionalist}\ (b < 1.0)\ \quad\vert\quad\ \mathbf{Pushover}\ (b = 1.0)\ \quad\vert\quad\ \mathbf{Opportunist}\ (b > 1.0)\ \longrightarrow$" "\n" r"Behavioral Memory Bias Parameter $b$", 
              fontsize=11.5, fontweight='bold', labelpad=10)
ax.set_ylabel(r"Panic Relaxation Half-Life $T_{1/2}$ (Trading Days to reach $n_+ = 0.50$)", fontsize=11.5, fontweight='bold')

ax.set_title(r"Macro Swarm Panic Relaxation Half-Life $T_{1/2}$ vs. Memory Weighting Bias $b$" "\n"
             r"(Calibrated on Indian Nifty 50 TRI Daily Returns, $N=150$ Agents, Noise $\eta = 0.30$, Peer Contagion $p_{\text{peer}} = 1.0$)",
             fontsize=12, fontweight='bold', pad=12)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='center right', fontsize=10, framealpha=0.95, edgecolor='gray', title=r"$\mathbf{Memory\ Horizon}$")

plt.tight_layout()

output_plot_path = os.path.join(OUTPUT_DIR, "thalf_vs_b_continuous_sweep.png")
fig.savefig(output_plot_path, dpi=250, bbox_inches='tight')
plt.close(fig)
print(f"✅ Saved continuous b-sweep relaxation plot: {output_plot_path}")
