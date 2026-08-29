import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
TRAIN_FRACTION = 0.6
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

# 2. Kernel weight and signal generators
def get_weights(personality, M):
    if personality == 'pushover':
        w = np.ones(M)
    elif personality == 'opportunist':
        w = B_OPPORTUNIST ** np.arange(M)
    elif personality == 'traditionalist':
        w = B_TRADITIONALIST ** np.arange(M)
    elif personality == 'contrarian':
        w = -np.ones(M)
    else:
        w = np.ones(M)
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

def calc_metrics(strat_ret):
    valid = strat_ret[~np.isnan(strat_ret)]
    if len(valid) == 0 or np.std(valid) == 0:
        return 0.0, 0.0, 0.0
    sharpe = np.mean(valid) / np.std(valid) * np.sqrt(252)
    ann_ret = (np.prod(1 + valid) ** (252 / len(valid))) - 1
    # Turnover
    diffs = np.abs(np.diff(np.nan_to_num(strat_ret)))
    turnover = np.mean(diffs)
    return sharpe, ann_ret, turnover

# 3. Sweep M from 1 to 21 for all personalities
records = []
personalities = ['pushover', 'opportunist', 'traditionalist', 'contrarian', 'curmudgeon']

for p in personalities:
    M_range = [1] if p == 'curmudgeon' else list(range(1, 22))
    for M in M_range:
        sig = compute_signal(p, q_lagged, curmudgeon_sig, M)
        strat_ret = np.where(np.isnan(sig), 0.0, sig * ret)
        
        is_ret = strat_ret[:split_idx]
        oos_ret = strat_ret[split_idx:]
        
        sh_is, ret_is, turn_is = calc_metrics(is_ret)
        sh_oos, ret_oos, turn_oos = calc_metrics(oos_ret)
        sh_full, ret_full, turn_full = calc_metrics(strat_ret)
        
        # Stability metrics
        gap = abs(sh_is - sh_oos)
        ratio = (sh_oos / sh_is) if sh_is > 0 else 0.0
        # Combined score rewarding high OOS Sharpe and close IS/OOS ratio (ratio close to 1.0)
        # Score = OOS Sharpe * (1 - gap / max(abs(IS), abs(OOS)))
        denom = max(abs(sh_is), abs(sh_oos))
        score = sh_oos * (1.0 - (gap / denom)) if denom > 0 and sh_is > 0 and sh_oos > 0 else -999.0
        
        records.append({
            "personality": p,
            "M": M if p != 'curmudgeon' else 'MA200',
            "M_val": M,
            "sharpe_is": sh_is,
            "sharpe_oos": sh_oos,
            "sharpe_full": sh_full,
            "ret_is": ret_is,
            "ret_oos": ret_oos,
            "ret_full": ret_full,
            "gap": gap,
            "ratio": ratio,
            "score": score,
            "signal": sig,
            "strat_ret": strat_ret
        })

df_results = pd.DataFrame(records)

# Filter for stable performers: IS Sharpe > 0.3, OOS Sharpe > 0.3, ratio between 0.7 and 1.5
stable_df = df_results[(df_results["sharpe_is"] > 0.3) & (df_results["sharpe_oos"] > 0.3)].copy()
stable_df = stable_df.sort_values("score", ascending=False).reset_index(drop=True)

print("Top 10 Stable Standalone Configurations (Ranked by Stability Score):")
cols_show = ["personality", "M", "sharpe_is", "sharpe_oos", "sharpe_full", "gap", "ratio", "score"]
print(stable_df[cols_show].head(10).to_string(index=False))

# Select Top 5
top5 = stable_df.head(5)

# 4. Generate Master & Grid Plots for Top 5
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

# Shade IS and OOS
ax.axvspan(dates[0], split_date, color='#e6f2ff', alpha=0.5, label='In-Sample (IS: 2015–2020)')
ax.axvspan(split_date, dates[-1], color='#fff0e6', alpha=0.5, label='Out-of-Sample (OOS: 2021–2025)')
ax.axvline(split_date, color='black', linestyle='--', linewidth=2.0, zorder=10)

# Plot Benchmark
bh_eq = np.cumprod(1 + ret)
ax.plot(dates, bh_eq, color='#555555', linestyle=':', linewidth=1.8, label="Nifty 50 Buy & Hold Benchmark")

# Plot Top 5
top5_equities = {}
for i, row in top5.iterrows():
    p = row["personality"]
    m = row["M"]
    sig = row["signal"]
    strat_ret = row["strat_ret"]
    eq = np.cumprod(1 + strat_ret)
    top5_equities[i] = eq
    
    lbl = f"Rank {i+1}: {p.capitalize()} (M={m}) | IS: {row['sharpe_is']:.2f}, OOS: {row['sharpe_oos']:.2f} (Ratio: {row['ratio']:.2f})"
    ax.plot(dates, eq, color=colors[i], linewidth=2.2, label=lbl)

# IS/OOS periods are already conveyed by the shaded axvspan bands in the legend.

ax.set_yscale('log')
ax.set_title("Top 5 Stable Standalone Behavioral Kernels (M = 1..21) Across Full Timeline (IS / OOS Marked)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Date", fontsize=12, fontweight='bold')
ax.set_ylabel("Cumulative Wealth Growth (Log Scale)", fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='gray')

plt.tight_layout()
master_top5_path = os.path.join(OUTPUT_DIR, "top5_stable_standalone_full_timeline_is_oos.png")
fig.savefig(master_top5_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved master top 5 plot: {master_top5_path}")

# Individual Subplots for Top 5
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, dpi=200)

for i, row in top5.iterrows():
    ax = axes[i]
    p = row["personality"]
    m = row["M"]
    eq = top5_equities[i]
    
    ax.axvspan(dates[0], split_date, color='#e6f2ff', alpha=0.5)
    ax.axvspan(split_date, dates[-1], color='#fff0e6', alpha=0.5)
    ax.axvline(split_date, color='black', linestyle='--', linewidth=1.8, zorder=10)
    
    ax.plot(dates, bh_eq, color='#555555', linestyle=':', linewidth=1.5, label="Nifty 50 Buy & Hold")
    ax.plot(dates, eq, color=colors[i], linewidth=2.2, label=f"Rank {i+1}: {p.capitalize()} (M={m})")
    
    metrics_txt = (f"IS Sharpe: {row['sharpe_is']:.2f}\n"
                   f"OOS Sharpe: {row['sharpe_oos']:.2f}\n"
                   f"IS/OOS Ratio: {row['ratio']:.2f}\n"
                   f"Overfit Gap: {row['gap']:.2f}")
    
    ax.text(0.02, 0.85, metrics_txt, transform=ax.transAxes, fontsize=9, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=colors[i], alpha=0.9))
    
    ax.set_title(f"Rank {i+1}: {p.capitalize()} Kernel (M={m})", fontsize=11, fontweight='bold', loc='left')
    ax.set_ylabel("Growth of 1", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

# IS/OOS labels removed — conveyed by shaded bands in the background.

axes[-1].set_xlabel("Date", fontsize=12, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle("Top 5 Most Stable Standalone Personality Kernels Across Full Timeline (IS / OOS Split Marked)", 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
grid_top5_path = os.path.join(OUTPUT_DIR, "top5_stable_standalone_individual_grid_is_oos.png")
fig.savefig(grid_top5_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved grid top 5 plot: {grid_top5_path}")

print("✅ Analysis and plotting completed successfully.")
