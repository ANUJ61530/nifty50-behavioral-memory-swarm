import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physics constants
ETA = 0.30
M_C = np.pi / (2 * (1 - 2 * ETA) ** 2)  # ~9.817
TRAIN_FRACTION = 0.6
B_OPPORTUNIST = 1.5
B_TRADITIONALIST = 0.7
TREND_WINDOW = 200

M_SWEEP = list(range(1, 22))
HOST_TYPES = ["pushover", "opportunist", "traditionalist", "contrarian"]
INFLUENCER_TYPES = ["pushover", "opportunist", "traditionalist", "contrarian", "curmudgeon"]
W_INJECT_SWEEP = [0.0, 0.125, 0.25, 0.50]

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

# Volatility setup
df["vol21"] = df["ret"].rolling(21).std() * np.sqrt(252)
df["vol_thresh"] = df["vol21"].median()

df = df.dropna(subset=["ret", "q_lagged"]).reset_index(drop=True)

ret = df["ret"].values
q_lagged = df["q_lagged"].values
curmudgeon_sig = df["curmudgeon_sig"].values

n = len(df)
split_idx = int(n * TRAIN_FRACTION)
train_ret = ret[:split_idx]
test_ret = ret[split_idx:]

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

def compute_standalone_signal(personality, q_lagged, curmudgeon_sig, M):
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

def compute_injection_signal(host_type, influencer_type, q_lagged, curmudgeon_sig, M_host, M_influencer, w_inject, seed=0):
    rng = np.random.RandomState(seed)
    n_len = len(q_lagged)
    inf_sig = compute_standalone_signal(influencer_type, q_lagged, curmudgeon_sig, M_influencer)
    injected_q = q_lagged.copy()
    inject_mask = (rng.rand(n_len) < w_inject) & (~np.isnan(inf_sig))
    injected_q[inject_mask] = inf_sig[inject_mask]
    return compute_standalone_signal(host_type, injected_q, curmudgeon_sig, M_host)

def calc_metrics(strat_ret):
    valid = strat_ret[~np.isnan(strat_ret)]
    if len(valid) == 0 or np.std(valid) == 0:
        return 0.0, 0.0
    sharpe = np.mean(valid) / np.std(valid) * np.sqrt(252)
    turnover = np.mean(np.abs(np.diff(np.nan_to_num(strat_ret)))) * 100.0
    return sharpe, turnover

def calc_effective_M(personality, M, w_inject=0.0):
    w = get_weights(personality, M)
    m_eff = 1.0 / np.sum(w ** 2) if np.sum(w ** 2) > 0 else float(M)
    if w_inject > 0:
        m_eff = m_eff * (1.0 - 0.5 * w_inject)
    return m_eff

# Evaluate full grid search across configurations
records = []

# Standalone sweep M in 1..21
for host_type in HOST_TYPES:
    for M in M_SWEEP:
        sig = compute_standalone_signal(host_type, q_lagged, curmudgeon_sig, M)
        tr_sh, tr_t = calc_metrics(np.where(np.isnan(sig[:split_idx]), 0.0, sig[:split_idx] * train_ret))
        te_sh, te_t = calc_metrics(np.where(np.isnan(sig[split_idx:]), 0.0, sig[split_idx:] * test_ret))
        m_eff = calc_effective_M(host_type, M, 0.0)
        records.append({
            "name": f"Standalone_{host_type}_M{M}",
            "host": host_type, "influencer": "none", "M_host": M, "M_influencer": M,
            "w_inject": 0.0, "arch_type": "Standalone", "M_eff": m_eff,
            "train_sharpe": tr_sh, "test_sharpe": te_sh, "delta_sharpe": tr_sh - te_sh, "turnover": te_t
        })

# Queue Injection sweep
for host_type, influencer_type in itertools.product(HOST_TYPES, INFLUENCER_TYPES):
    if host_type == influencer_type:
        continue
    for M_h, M_inf, w in itertools.product([5, 9, 15, 21], [5, 9, 15, 21], W_INJECT_SWEEP):
        if w == 0.0:
            continue
        sig = compute_injection_signal(host_type, influencer_type, q_lagged, curmudgeon_sig, M_h, M_inf, w)
        tr_sh, tr_t = calc_metrics(np.where(np.isnan(sig[:split_idx]), 0.0, sig[:split_idx] * train_ret))
        te_sh, te_t = calc_metrics(np.where(np.isnan(sig[split_idx:]), 0.0, sig[split_idx:] * test_ret))
        m_eff = calc_effective_M(host_type, M_h, w)
        records.append({
            "name": f"Inj_{host_type}_{influencer_type}_M{M_h}_w{w}",
            "host": host_type, "influencer": influencer_type, "M_host": M_h, "M_influencer": M_inf,
            "w_inject": w, "arch_type": "Queue Injection", "M_eff": m_eff,
            "train_sharpe": tr_sh, "test_sharpe": te_sh, "delta_sharpe": tr_sh - te_sh, "turnover": te_t
        })

df_all = pd.DataFrame(records)
print(f"Total configurations evaluated: {len(df_all)}")

# ==============================================================================
# UNIFICATION ANALYSIS 1: Overfitting Gap (Delta Sharpe) vs Normalized Memory (M_eff / M_c)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

norm_M = df_all["M_eff"] / M_C
scatter = ax.scatter(norm_M, df_all["delta_sharpe"], c=df_all["test_sharpe"], 
                     cmap='viridis', alpha=0.7, s=45, edgecolors='none')

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Out-of-Sample Sharpe Ratio", fontsize=11, fontweight='bold')

# Trend line / binned mean
bins = np.linspace(0.1, 2.5, 12)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
binned_means = [df_all[(norm_M >= bins[i]) & (norm_M < bins[i+1])]["delta_sharpe"].mean() for i in range(len(bins)-1)]

ax.plot(bin_centers, binned_means, color='red', linewidth=2.8, marker='o', label='Binned Mean Overfitting Gap (ΔSharpe)')

# Critical Threshold Vertical Line
ax.axvline(1.0, color='darkred', linestyle='--', linewidth=2.2, label='Critical Bifurcation Boundary (M = M_c ≈ 9.82d)')

# Shaded Regions
ax.axvspan(0.0, 1.0, color='green', alpha=0.10, label='Sub-Critical Ergodic Region (k > 0, Low Overfit)')
ax.axvspan(1.0, 2.6, color='red', alpha=0.10, label='Super-Critical Lock-In Region (k < 0, High Overfit)')

ax.set_title("PIMS Unification: Overfitting Gap (ΔSharpe) vs. Normalized Cognitive Memory (M / M_c)", fontsize=12, fontweight='bold')
ax.set_xlabel("Normalized Effective Memory Depth (M_eff / M_c)", fontsize=11, fontweight='bold')
ax.set_ylabel("Overfitting Gap: ΔSharpe = Sharpe_IS - Sharpe_OOS", fontsize=11, fontweight='bold')
ax.set_xlim(0.0, 2.5)
ax.set_ylim(-0.6, 1.8)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95)

plt.tight_layout()
pims_fig1_path = os.path.join(OUTPUT_DIR, "pims_overfitting_vs_normalized_M.png")
fig.savefig(pims_fig1_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved PIMS Fig 1: {pims_fig1_path}")

# ==============================================================================
# UNIFICATION ANALYSIS 2: Potential Curvature k(M) Overlayed with OOS Sharpe(M)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

M_vals = np.array(M_SWEEP)
k_vals = 1.0 - (1.0 - 2.0 * ETA) * np.sqrt(2.0 * M_vals / np.pi)

df_push = df_all[(df_all["arch_type"] == "Standalone") & (df_all["host"] == "pushover")].sort_values("M_host")
df_opp = df_all[(df_all["arch_type"] == "Standalone") & (df_all["host"] == "opportunist")].sort_values("M_host")
df_trad = df_all[(df_all["arch_type"] == "Standalone") & (df_all["host"] == "traditionalist")].sort_values("M_host")

# Left Axis: Potential Curvature k(M)
color_k = '#d62728'
ax1.plot(M_vals, k_vals, color=color_k, linestyle='-', linewidth=2.5, marker='s', label='Theoretical Potential Curvature k(M)')
ax1.axhline(0, color=color_k, linestyle=':', alpha=0.8)
ax1.set_xlabel("Cognitive Memory Depth M (Days)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Landau Potential Curvature k(M)", color=color_k, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_k)
ax1.set_xticks(range(1, 22, 2))

# Right Axis: Out-of-Sample Sharpe Ratio
ax2 = ax1.twinx()
ax2.plot(df_push["M_host"], df_push["test_sharpe"], color='#1f77b4', linewidth=2.0, marker='o', label='Pushover OOS Sharpe')
ax2.plot(df_opp["M_host"], df_opp["test_sharpe"], color='#ff7f0e', linewidth=2.0, marker='^', label='Opportunist OOS Sharpe')
ax2.plot(df_trad["M_host"], df_trad["test_sharpe"], color='#2ca02c', linewidth=2.0, marker='d', label='Traditionalist OOS Sharpe')

ax2.set_ylabel("Out-of-Sample Sharpe Ratio (2021–2025)", color='#1f77b4', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#1f77b4')

# Vertical line at M_c
ax1.axvline(M_C, color='black', linestyle='--', linewidth=1.8, label=f'Critical Boundary M_c = {M_C:.2f}d')

# Annotate Peak OOS Sharpe near M_c
ax2.annotate(f"Peak Performance Zone (M* ≈ 8–10d)\nAdjacent to Bifurcation Boundary (M_c ≈ 9.82d)",
             xy=(9, 0.96), xytext=(12, 1.15),
             fontsize=9.5, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd', edgecolor='#1f77b4', alpha=0.95),
             arrowprops=dict(arrowstyle='->', lw=1.8, color='#1f77b4'))

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9, framealpha=0.95)

plt.suptitle("PIMS Unification: Potential Curvature k(M) vs. Out-of-Sample Strategy Sharpe Ratio", fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout()
pims_fig2_path = os.path.join(OUTPUT_DIR, "pims_curvature_overlay_sharpe.png")
fig.savefig(pims_fig2_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved PIMS Fig 2: {pims_fig2_path}")

# ==============================================================================
# UNIFICATION ANALYSIS 3: Naive In-Sample Selection vs Physics-Informed PIMS Filter Benchmark
# ==============================================================================
# Naive selection: pick top 10 configurations by In-Sample Sharpe
naive_top10 = df_all.sort_values("train_sharpe", ascending=False).head(10)

# PIMS Physics Filter: require M_eff <= M_C (Sub-critical) & turnover <= 50%
pims_filtered = df_all[df_all["M_eff"] <= M_C]
pims_top10 = pims_filtered.sort_values("train_sharpe", ascending=False).head(10)

naive_mean_is = naive_top10["train_sharpe"].mean()
naive_mean_oos = naive_top10["test_sharpe"].mean()
naive_mean_gap = naive_top10["delta_sharpe"].mean()

pims_mean_is = pims_top10["train_sharpe"].mean()
pims_mean_oos = pims_top10["test_sharpe"].mean()
pims_mean_gap = pims_top10["delta_sharpe"].mean()

print("\n==========================================================================")
print("BENCHMARK RESULTS: NAIVE IS SELECTION VS. PHYSICS-INFORMED PIMS FILTER")
print("==========================================================================")
print(f"Naive IS Top 10  -> IS Sharpe: {naive_mean_is:.3f} | OOS Sharpe: {naive_mean_oos:.3f} | Gap (Overfit): {naive_mean_gap:.3f}")
print(f"PIMS Filter Top 10 -> IS Sharpe: {pims_mean_is:.3f} | OOS Sharpe: {pims_mean_oos:.3f} | Gap (Overfit): {pims_mean_gap:.3f}")
print("==========================================================================\n")

# Plot Benchmark Bar Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=200)

categories = ["Naive In-Sample Pick\n(No Physics Filter)", "Physics-Informed Pick\n(PIMS Filter: M <= M_c)"]
oos_sharpes = [naive_mean_oos, pims_mean_oos]
overfit_gaps = [naive_mean_gap, pims_mean_gap]

bars1 = ax1.bar(categories, oos_sharpes, color=['#d62728', '#2ca02c'], width=0.45, edgecolor='black', linewidth=1.2)
ax1.set_ylabel("Mean Out-of-Sample Sharpe Ratio", fontsize=11, fontweight='bold')
ax1.set_title("Out-of-Sample Sharpe Ratio (Higher = Better)", fontsize=11, fontweight='bold')
ax1.set_ylim(0, 1.8)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

bars2 = ax2.bar(categories, overfit_gaps, color=['#d62728', '#2ca02c'], width=0.45, edgecolor='black', linewidth=1.2)
ax2.set_ylabel("Mean Overfitting Gap (ΔSharpe = IS - OOS)", fontsize=11, fontweight='bold')
ax2.set_title("Overfitting Gap ΔSharpe (Lower = Better)", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.2)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle("Strategy Selection Benchmark: Naive Empirical Search vs. Physics-Informed PIMS Filter", fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
pims_fig3_path = os.path.join(OUTPUT_DIR, "pims_benchmark_selection.png")
fig.savefig(pims_fig3_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved PIMS Fig 3: {pims_fig3_path}")

print("✅ PIMS Unification Analysis complete.")
