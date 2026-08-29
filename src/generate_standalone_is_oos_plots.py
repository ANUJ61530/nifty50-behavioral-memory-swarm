import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
TRAIN_FRACTION = 0.6
M = 21
B_OPPORTUNIST = 1.5
B_TRADITIONALIST = 0.7
TREND_WINDOW = 200

# 1. Load Data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["ret"] = df["close"].pct_change()
df["q"] = np.sign(df["ret"])
df.loc[df["q"] == 0, "q"] = 1.0
df["q_lagged"] = df["q"].shift(1)

# Curmudgeon signal setup
df["ma200"] = df["close"].rolling(window=TREND_WINDOW).mean()
df["d_ma200"] = df["ma200"].diff()
df["curmudgeon_sig"] = np.sign(df["d_ma200"].shift(1))
df["curmudgeon_sig"] = df["curmudgeon_sig"].fillna(1.0)
df.loc[df["curmudgeon_sig"] == 0, "curmudgeon_sig"] = 1.0

df = df.dropna(subset=["ret", "q_lagged"]).reset_index(drop=True)

dates = df["timestamp"].values
ret = df["ret"].values
q_lagged = df["q_lagged"].values
curmudgeon_sig = df["curmudgeon_sig"].values

n = len(df)
split_idx = int(n * TRAIN_FRACTION)
split_date = dates[split_idx]

# 2. Kernel weights & signal calculations
def get_weights(personality, M):
    if personality == 'pushover':
        w = np.ones(M)
    elif personality == 'opportunist':
        w = B_OPPORTUNIST ** np.arange(M)
    elif personality == 'traditionalist':
        w = B_TRADITIONALIST ** np.arange(M)
    elif personality == 'contrarian':
        w = -np.ones(M)
    s = np.sum(np.abs(w))
    return w / s if s > 0 else w

def compute_signal(personality, q_lagged, curmudgeon_sig, M):
    if personality == 'curmudgeon':
        return curmudgeon_sig
    w = get_weights(personality, M)
    n_len = len(q_lagged)
    out = np.full(n_len, np.nan)
    if n_len >= M:
        windows = np.lib.stride_tricks.sliding_window_view(q_lagged, M)
        sigs = np.dot(windows, w)
        signs = np.sign(sigs)
        signs[signs == 0] = 1.0
        out[M-1:] = signs
    return out

def calc_sharpe(strat_ret):
    valid = strat_ret[~np.isnan(strat_ret)]
    if len(valid) == 0 or np.std(valid) == 0:
        return 0.0
    return np.mean(valid) / np.std(valid) * np.sqrt(252)

personalities = ['pushover', 'opportunist', 'traditionalist', 'contrarian', 'curmudgeon']
colors = {
    'pushover': '#1f77b4',       # Royal Blue
    'opportunist': '#ff7f0e',      # Orange
    'traditionalist': '#2ca02c',   # Green
    'contrarian': '#d62728',       # Red
    'curmudgeon': '#9467bd',       # Purple
    'buy_hold': '#555555'          # Dark Gray
}

labels = {
    'pushover': 'Pushover (Flat Momentum)',
    'opportunist': 'Opportunist (Recency Weighted)',
    'traditionalist': 'Traditionalist (Distance Weighted)',
    'contrarian': 'Contrarian (Mean Reversion)',
    'curmudgeon': 'Curmudgeon (MA200 Structural Anchor)',
    'buy_hold': 'Nifty 50 Buy & Hold Benchmark'
}

# Compute equity curves for all
signals = {}
equities = {}
sharpe_is = {}
sharpe_oos = {}
sharpe_total = {}

# Benchmark
bh_ret = ret
bh_eq = np.cumprod(1 + bh_ret)
equities['buy_hold'] = bh_eq
sharpe_is['buy_hold'] = calc_sharpe(bh_ret[:split_idx])
sharpe_oos['buy_hold'] = calc_sharpe(bh_ret[split_idx:])
sharpe_total['buy_hold'] = calc_sharpe(bh_ret)

for p in personalities:
    sig = compute_signal(p, q_lagged, curmudgeon_sig, M)
    signals[p] = sig
    strat_ret = np.where(np.isnan(sig), 0.0, sig * ret)
    eq = np.cumprod(1 + strat_ret)
    equities[p] = eq
    sharpe_is[p] = calc_sharpe(strat_ret[:split_idx])
    sharpe_oos[p] = calc_sharpe(strat_ret[split_idx:])
    sharpe_total[p] = calc_sharpe(strat_ret)

# ==============================================================================
# PLOT 1: Master Combined Plot (IS + OOS Marked)
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

# Shade IS and OOS regions
ax.axvspan(dates[0], split_date, color='#e6f2ff', alpha=0.5, label='In-Sample (IS: 2015–2020)')
ax.axvspan(split_date, dates[-1], color='#fff0e6', alpha=0.5, label='Out-of-Sample (OOS: 2021–2025)')

# Vertical split line
ax.axvline(split_date, color='black', linestyle='--', linewidth=2.0, zorder=10)

# Plot equity curves
ax.plot(dates, equities['buy_hold'], color=colors['buy_hold'], linestyle=':', linewidth=1.8, label=f"{labels['buy_hold']} (OOS Sharpe: {sharpe_oos['buy_hold']:.2f})")

for p in personalities:
    lbl = f"{labels[p]} | IS: {sharpe_is[p]:.2f}, OOS: {sharpe_oos[p]:.2f}"
    ax.plot(dates, equities[p], color=colors[p], linewidth=2.2, label=lbl)

ax.set_yscale('log')
ax.set_title("5 Standalone Behavioral Memory Personalities: Complete Timeline (In-Sample + Out-of-Sample Marked)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Date", fontsize=12, fontweight='bold')
ax.set_ylabel("Cumulative Wealth Growth (Log Scale)", fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95, edgecolor='gray')

plt.tight_layout()
master_plot_path = os.path.join(OUTPUT_DIR, "standalone_5_personalities_full_timeline_is_oos.png")
fig.savefig(master_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved master plot: {master_plot_path}")

# ==============================================================================
# PLOT 2: 5-Panel Grid Subplot (Each Personality Individual IS vs OOS Marked)
# ==============================================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, dpi=200)

for i, p in enumerate(personalities):
    ax = axes[i]
    
    # Shade IS and OOS
    ax.axvspan(dates[0], split_date, color='#e6f2ff', alpha=0.5)
    ax.axvspan(split_date, dates[-1], color='#fff0e6', alpha=0.5)
    ax.axvline(split_date, color='black', linestyle='--', linewidth=1.8, zorder=10)
    
    # Plot Benchmark and Personality
    ax.plot(dates, equities['buy_hold'], color=colors['buy_hold'], linestyle=':', linewidth=1.5, label="Nifty 50 Buy & Hold")
    ax.plot(dates, equities[p], color=colors[p], linewidth=2.2, label=labels[p])
    
    # Metrics box
    metrics_txt = (f"IS Sharpe: {sharpe_is[p]:.2f}\n"
                   f"OOS Sharpe: {sharpe_oos[p]:.2f}\n"
                   f"Full Sharpe: {sharpe_total[p]:.2f}")
    
    ax.text(0.02, 0.85, metrics_txt, transform=ax.transAxes, fontsize=9, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=colors[p], alpha=0.9))
    
    ax.set_title(f"Kernel {i+1}: {labels[p]} (M={M})", fontsize=11, fontweight='bold', loc='left')
    ax.set_ylabel("Growth of 1", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

# Labels for top IS and OOS
axes[0].text(dates[int(split_idx * 0.5)], axes[0].get_ylim()[1] * 0.9, "◄ IN-SAMPLE (2015–2020) ►", 
             fontsize=10, fontweight='bold', ha='center', color='#1f77b4')
axes[0].text(dates[split_idx + int((n - split_idx) * 0.5)], axes[0].get_ylim()[1] * 0.9, "◄ OUT-OF-SAMPLE (2021–2025) ►", 
             fontsize=10, fontweight='bold', ha='center', color='#d95f02')

axes[-1].set_xlabel("Date", fontsize=12, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle("Individual Subplots: 5 Standalone Personality Kernels Across Complete Timeline (IS / OOS Split Marked)", 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
grid_plot_path = os.path.join(OUTPUT_DIR, "standalone_5_personalities_individual_grid_is_oos.png")
fig.savefig(grid_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved grid plot: {grid_plot_path}")

print("✅ Standalone IS/OOS plots generation complete.")
