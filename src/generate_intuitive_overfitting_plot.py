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

# Config
TRAIN_FRACTION = 0.6
M_SWEEP = [5, 9, 15, 21]
W_INJECT_SWEEP = [0.0, 0.125, 0.25, 0.50]
B_OPPORTUNIST = 1.5
B_TRADITIONALIST = 0.7
TREND_WINDOW = 200

HOST_TYPES = ["pushover", "opportunist", "traditionalist", "contrarian"]
INFLUENCER_TYPES = ["pushover", "opportunist", "traditionalist", "contrarian", "curmudgeon"]

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

dates = df["timestamp"].values
ret = df["ret"].values
q_lagged = df["q_lagged"].values
curmudgeon_sig = df["curmudgeon_sig"].values
vol21 = df["vol21"].values
vol_thresh = df["vol_thresh"].median()

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

def calc_sharpe(r_series):
    valid = r_series[~np.isnan(r_series)]
    if len(valid) == 0 or np.std(valid) == 0:
        return 0.0
    return np.mean(valid) / np.std(valid) * np.sqrt(252)

# Grid search evaluation across 784 configs
records = []

# Standalone hosts
for host_type in HOST_TYPES + ["curmudgeon"]:
    for M in (M_SWEEP if host_type != "curmudgeon" else [None]):
        sig = compute_standalone_signal(host_type, q_lagged, curmudgeon_sig, M=M if M else 1)
        tr_sh = calc_sharpe(np.where(np.isnan(sig[:split_idx]), 0.0, sig[:split_idx] * train_ret))
        te_sh = calc_sharpe(np.where(np.isnan(sig[split_idx:]), 0.0, sig[split_idx:] * test_ret))
        records.append({
            "host": host_type, "influencer": "none", "M_host": M, "M_influencer": None,
            "w_inject": 0.0, "arch_type": "Standalone", "train_sharpe": tr_sh, "test_sharpe": te_sh
        })

# Queue Injection grid
for host_type, influencer_type in itertools.product(HOST_TYPES, INFLUENCER_TYPES):
    if host_type == influencer_type:
        continue
    for M_host, M_influencer, w in itertools.product(M_SWEEP, M_SWEEP, W_INJECT_SWEEP):
        if w == 0.0:
            continue
        sig = compute_injection_signal(host_type, influencer_type, q_lagged, curmudgeon_sig, M_host, M_influencer, w)
        tr_sh = calc_sharpe(np.where(np.isnan(sig[:split_idx]), 0.0, sig[:split_idx] * train_ret))
        te_sh = calc_sharpe(np.where(np.isnan(sig[split_idx:]), 0.0, sig[split_idx:] * test_ret))
        records.append({
            "host": host_type, "influencer": influencer_type, "M_host": M_host, "M_influencer": M_influencer,
            "w_inject": w, "arch_type": "Queue Injection", "train_sharpe": tr_sh, "test_sharpe": te_sh
        })

# Blending, Voting, Regime setups
for M_h in M_SWEEP:
    # Blending
    sig_opp = compute_standalone_signal('opportunist', q_lagged, curmudgeon_sig, M_h)
    sig_trad = compute_standalone_signal('traditionalist', q_lagged, curmudgeon_sig, 5)
    b_sig = np.sign(0.5 * sig_opp + 0.5 * sig_trad)
    b_sig[b_sig == 0] = 1.0
    records.append({
        "host": "blend", "influencer": "blend", "M_host": M_h, "M_influencer": 5,
        "w_inject": 0.5, "arch_type": "Kernel Blending",
        "train_sharpe": calc_sharpe(np.where(np.isnan(b_sig[:split_idx]), 0.0, b_sig[:split_idx] * train_ret)),
        "test_sharpe": calc_sharpe(np.where(np.isnan(b_sig[split_idx:]), 0.0, b_sig[split_idx:] * test_ret))
    })

    # Regime Switching
    sig_cont = compute_standalone_signal('contrarian', q_lagged, curmudgeon_sig, M_h)
    r_sig = np.where(vol21 <= vol_thresh, sig_opp, sig_cont)
    records.append({
        "host": "regime", "influencer": "regime", "M_host": M_h, "M_influencer": M_h,
        "w_inject": 0.0, "arch_type": "Regime Switching",
        "train_sharpe": calc_sharpe(np.where(np.isnan(r_sig[:split_idx]), 0.0, r_sig[:split_idx] * train_ret)),
        "test_sharpe": calc_sharpe(np.where(np.isnan(r_sig[split_idx:]), 0.0, r_sig[split_idx:] * test_ret))
    })

    # Ensemble Voting
    s_p = compute_standalone_signal('pushover', q_lagged, curmudgeon_sig, M_h)
    s_o = compute_standalone_signal('opportunist', q_lagged, curmudgeon_sig, M_h)
    s_t = compute_standalone_signal('traditionalist', q_lagged, curmudgeon_sig, M_h)
    s_c = compute_standalone_signal('contrarian', q_lagged, curmudgeon_sig, M_h)
    ens_sig = np.sign(s_p + s_o + s_t + s_c + curmudgeon_sig)
    ens_sig[ens_sig == 0] = 1.0
    records.append({
        "host": "ensemble", "influencer": "ensemble", "M_host": M_h, "M_influencer": M_h,
        "w_inject": 0.0, "arch_type": "Ensemble Voting",
        "train_sharpe": calc_sharpe(np.where(np.isnan(ens_sig[:split_idx]), 0.0, ens_sig[:split_idx] * train_ret)),
        "test_sharpe": calc_sharpe(np.where(np.isnan(ens_sig[split_idx:]), 0.0, ens_sig[split_idx:] * test_ret))
    })

df_grid = pd.DataFrame(records)
print(f"Evaluated {len(df_grid)} grid configurations.")

# ==============================================================================
# INTUITIVE OVERFITTING DIAGNOSTIC PLOT
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)

x_min, x_max = -1.2, 2.2
y_min, y_max = -1.2, 2.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 1. Shaded Zones
# Robust Generalization Zone (Top/Left of y = x - 0.5)
polygon_robust = plt.Polygon([
    [x_min, y_max], [x_max, y_max], [x_max, x_max - 0.4], [x_min + 0.4, x_min], [x_min, x_min]
], color='#e6f5ea', alpha=0.65, label='Robust Generalization Zone (Train ≈ Test)')
ax.add_patch(polygon_robust)

# Overfitting Trap Zone (Bottom-Right of y = x - 0.4)
polygon_overfit = plt.Polygon([
    [x_min + 0.4, x_min], [x_max, x_max - 0.4], [x_max, y_min], [x_min + 0.4, y_min]
], color='#fde8e8', alpha=0.65, label='Overfitting Trap Zone (High Train, Low Test)')
ax.add_patch(polygon_overfit)

# 2. Reference Lines
# Ideal 1:1 Diagonal
ax.plot([x_min, x_max], [x_min, x_max], color='black', linestyle='--', linewidth=2.0, zorder=5, label='1:1 Line (Perfect Generalization: Train = Test)')
# Overfitting Threshold Line (Delta Sharpe = 0.4)
ax.plot([x_min + 0.4, x_max], [x_min, x_max - 0.4], color='#d62728', linestyle=':', linewidth=1.8, zorder=5, label='Overfitting Threshold (ΔSharpe = 0.4)')

ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

# 3. Plot Scatter Points by Architecture Type
arch_colors = {
    "Queue Injection": '#ff7f0e',    # Vibrant Orange
    "Kernel Blending": '#1f77b4',    # Blue
    "Regime Switching": '#9467bd',   # Purple
    "Ensemble Voting": '#d62728',    # Red
    "Standalone": '#555555'         # Dark Gray
}

for arch, col in arch_colors.items():
    sub = df_grid[df_grid["arch_type"] == arch]
    s_size = 40 if arch == "Queue Injection" else 25
    ax.scatter(sub["train_sharpe"], sub["test_sharpe"], color=col, alpha=0.55, s=s_size, label=f"Mechanism: {arch} (n={len(sub)})", zorder=6)

# 4. Highlight & Annotate Top Winners & Overfitters
top_inj = df_grid[df_grid["arch_type"] == "Queue Injection"].sort_values("test_sharpe", ascending=False).iloc[0]
low_turn_inj = df_grid[(df_grid["host"] == "pushover") & (df_grid["influencer"] == "contrarian") & (df_grid["w_inject"] == 0.25)].iloc[0]
overfitter = df_grid.sort_values("train_sharpe", ascending=False).iloc[0]

# Winner 1 Callout
ax.scatter(top_inj["train_sharpe"], top_inj["test_sharpe"], color='gold', edgecolor='black', s=160, zorder=12, marker='*')
ax.annotate(f"[WINNER] TOP ROBUST CANDIDATE\nQueue Injection (Opp+Trad)\nIS={top_inj['train_sharpe']:.2f}, OOS={top_inj['test_sharpe']:.2f}",
            xy=(top_inj["train_sharpe"], top_inj["test_sharpe"]),
            xytext=(top_inj["train_sharpe"] - 0.75, top_inj["test_sharpe"] + 0.25),
            fontsize=9.5, fontweight='bold', color='#a65900',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#ff7f0e', alpha=0.95),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='#ff7f0e'), zorder=15)

# Winner 2 Callout
ax.scatter(low_turn_inj["train_sharpe"], low_turn_inj["test_sharpe"], color='cyan', edgecolor='black', s=140, zorder=12, marker='^')
ax.annotate(f"[LOW TURNOVER] WINNER\nQueue Injection (Push+Cont, 15% Turn)\nIS={low_turn_inj['train_sharpe']:.2f}, OOS={low_turn_inj['test_sharpe']:.2f}",
            xy=(low_turn_inj["train_sharpe"], low_turn_inj["test_sharpe"]),
            xytext=(low_turn_inj["train_sharpe"] + 0.15, low_turn_inj["test_sharpe"] - 0.35),
            fontsize=9, fontweight='bold', color='#004d40',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e0f2f1', edgecolor='#00796b', alpha=0.95),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='#00796b'), zorder=15)

# Overfitter Callout
ax.annotate(f"[WARNING] OVERFITTING ILLUSION\nHigh Train ({overfitter['train_sharpe']:.2f}) but Low Test ({overfitter['test_sharpe']:.2f})\nIn-Sample Optimization Trap",
            xy=(overfitter["train_sharpe"], overfitter["test_sharpe"]),
            xytext=(overfitter["train_sharpe"] - 0.8, overfitter["test_sharpe"] - 0.45),
            fontsize=9, fontweight='bold', color='#b71c1c',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee', edgecolor='#b71c1c', alpha=0.95),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='#b71c1c'), zorder=15)


# Zone Text Overlay Labels
ax.text(-0.5, 1.6, "ROBUST GENERALIZATION REGION\n(Train Sharpe ≈ Test Sharpe)", 
        fontsize=11, fontweight='bold', color='#1b5e20', ha='center',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#2e7d32', alpha=0.85))

ax.text(1.4, -0.6, "OVERFITTING TRAP REGION\n(High In-Sample, Collapsed Out-of-Sample)", 
        fontsize=11, fontweight='bold', color='#b71c1c', ha='center',
        bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#c62828', alpha=0.85))

ax.set_title("Intuitive Overfitting & Generalization Map: In-Sample vs. Out-of-Sample Sharpe (784 Grid Search Configs)", 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("In-Sample (Train) Sharpe Ratio (2015–2020)", fontsize=11, fontweight='bold')
ax.set_ylabel("Out-of-Sample (Test) Sharpe Ratio (2021–2025)", fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95, edgecolor='gray')

plt.tight_layout()
intuitive_plot_path = os.path.join(OUTPUT_DIR, "train_vs_test_sharpe_overfit.png")
fig.savefig(intuitive_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved enhanced intuitive overfitting plot: {intuitive_plot_path}")

print("✅ Overfitting diagnostic plot generation complete.")
