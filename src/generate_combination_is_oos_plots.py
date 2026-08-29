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

# Volatility setup for regime switching
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
split_date = dates[split_idx]

# Signal building functions
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
    
    # Precompute influencer signal
    inf_sig = compute_standalone_signal(influencer_type, q_lagged, curmudgeon_sig, M_influencer)
    
    # Injected bit series
    injected_q = q_lagged.copy()
    inject_mask = (rng.rand(n_len) < w_inject) & (~np.isnan(inf_sig))
    injected_q[inject_mask] = inf_sig[inject_mask]
    
    # Evaluate host over injected queue
    return compute_standalone_signal(host_type, injected_q, curmudgeon_sig, M_host)

def calc_sharpe(strat_ret):
    valid = strat_ret[~np.isnan(strat_ret)]
    if len(valid) == 0 or np.std(valid) == 0:
        return 0.0
    return np.mean(valid) / np.std(valid) * np.sqrt(252)

# Build Combination Signals
signals = {}

# 1. Queue Injection (Opp M=9 + Trad M=5, w=0.5)
signals['inj_opp_trad'] = compute_injection_signal('opportunist', 'traditionalist', q_lagged, curmudgeon_sig, M_host=9, M_influencer=5, w_inject=0.5)

# 2. Queue Injection (Push M=21 + Cont M=21, w=0.25)
signals['inj_push_cont'] = compute_injection_signal('pushover', 'contrarian', q_lagged, curmudgeon_sig, M_host=21, M_influencer=21, w_inject=0.25)

# 3. Kernel Blending (Opp + Trad Convex Mix)
sig_opp = compute_standalone_signal('opportunist', q_lagged, curmudgeon_sig, 9)
sig_trad = compute_standalone_signal('traditionalist', q_lagged, curmudgeon_sig, 5)
blend_sig = np.sign(0.5 * sig_opp + 0.5 * sig_trad)
blend_sig[blend_sig == 0] = 1.0
signals['blend_opp_trad'] = blend_sig

# 4. Regime Switching (Vol Gated)
sig_cont = compute_standalone_signal('contrarian', q_lagged, curmudgeon_sig, 21)
regime_sig = np.where(vol21 <= vol_thresh, sig_opp, sig_cont)
signals['regime_switch'] = regime_sig

# 5. Ensemble Voting (Majority Rule across 5 standalone M=21)
s_push = compute_standalone_signal('pushover', q_lagged, curmudgeon_sig, 21)
s_opp = compute_standalone_signal('opportunist', q_lagged, curmudgeon_sig, 21)
s_trad = compute_standalone_signal('traditionalist', q_lagged, curmudgeon_sig, 21)
s_cont = compute_standalone_signal('contrarian', q_lagged, curmudgeon_sig, 21)
s_curm = curmudgeon_sig
ens_vote = np.sign(s_push + s_opp + s_trad + s_cont + s_curm)
ens_vote[ens_vote == 0] = 1.0
signals['ensemble_vote'] = ens_vote

# Evaluate Equities & Metrics
combo_names = ['inj_opp_trad', 'inj_push_cont', 'blend_opp_trad', 'regime_switch', 'ensemble_vote']
combo_labels = {
    'inj_opp_trad': 'Queue Injection (Opp M=9 + Trad M=5, w=0.5)',
    'inj_push_cont': 'Queue Injection (Push M=21 + Cont M=21, w=0.25)',
    'blend_opp_trad': 'Kernel Blending (Opp + Trad Convex Mix)',
    'regime_switch': 'Regime Switching (Vol Gated Opp/Cont)',
    'ensemble_vote': 'Ensemble Voting (Majority Rule)'
}

combo_colors = {
    'inj_opp_trad': '#ff7f0e',    # Orange
    'inj_push_cont': '#2ca02c',   # Green
    'blend_opp_trad': '#1f77b4',  # Blue
    'regime_switch': '#9467bd',  # Purple
    'ensemble_vote': '#d62728',   # Red
    'buy_hold': '#555555'         # Gray
}

equities = {}
sharpe_is = {}
sharpe_oos = {}
sharpe_full = {}

bh_eq = np.cumprod(1 + ret)
equities['buy_hold'] = bh_eq

for name in combo_names:
    sig = signals[name]
    s_ret = np.where(np.isnan(sig), 0.0, sig * ret)
    eq = np.cumprod(1 + s_ret)
    equities[name] = eq
    sharpe_is[name] = calc_sharpe(s_ret[:split_idx])
    sharpe_oos[name] = calc_sharpe(s_ret[split_idx:])
    sharpe_full[name] = calc_sharpe(s_ret)

# Generate Plot: Full Timeline IS + OOS Marked for Combination Architectures
fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

ax.axvspan(dates[0], split_date, color='#e6f2ff', alpha=0.5, label='In-Sample (IS: 2015–2020)')
ax.axvspan(split_date, dates[-1], color='#fff0e6', alpha=0.5, label='Out-of-Sample (OOS: 2021–2025)')
ax.axvline(split_date, color='black', linestyle='--', linewidth=2.0, zorder=10)

ax.plot(dates, bh_eq, color='#555555', linestyle=':', linewidth=1.8, label="Nifty 50 Buy & Hold Benchmark")

for name in combo_names:
    lbl = f"{combo_labels[name]} | IS: {sharpe_is[name]:.2f}, OOS: {sharpe_oos[name]:.2f}"
    ax.plot(dates, equities[name], color=combo_colors[name], linewidth=2.2, label=lbl)

ax.set_yscale('log')
ax.set_title("Personality Combination Architectures: Complete Timeline (In-Sample + Out-of-Sample Marked)", 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Date", fontsize=12, fontweight='bold')
ax.set_ylabel("Cumulative Wealth Growth (Log Scale)", fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.95, edgecolor='gray')

plt.tight_layout()
combo_plot_path = os.path.join(OUTPUT_DIR, "combination_architectures_full_timeline_is_oos.png")
fig.savefig(combo_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved combination plot: {combo_plot_path}")
