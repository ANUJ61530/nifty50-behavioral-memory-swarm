import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "nifty50_tri.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FRACTION   = 0.60
M_SWEEP          = list(range(1, 22))
B_OPPORTUNIST    = 1.5
B_TRADITIONALIST = 0.7
TREND_WINDOW     = 200

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["ret"]       = df["close"].pct_change()
df["q"]         = np.sign(df["ret"]).replace(0, 1.0)
df["q_lagged"]  = df["q"].shift(1)

df["ma200"]       = df["close"].rolling(TREND_WINDOW).mean()
df["d_ma200"]     = df["ma200"].diff()
df["curm_sig"]    = np.sign(df["d_ma200"].shift(1)).fillna(1.0).replace(0, 1.0)

df["vol21"]    = df["ret"].rolling(21).std() * np.sqrt(252)
vol_thresh     = df["vol21"].median()

df = df.dropna(subset=["ret", "q_lagged"]).reset_index(drop=True)

ret        = df["ret"].values
q_lagged   = df["q_lagged"].values
curm_sig   = df["curm_sig"].values
vol21      = df["vol21"].values
n          = len(df)
split_idx  = int(n * TRAIN_FRACTION)

# ── Weight & signal helpers ───────────────────────────────────────────────────
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

def compute_signal(personality, M):
    """Returns a position series (+1 / -1); NaN for warm-up period."""
    if personality == 'curmudgeon':
        return curm_sig.copy()
    w     = get_weights(personality, M)
    out   = np.full(n, np.nan)
    if n >= M:
        wins  = np.lib.stride_tricks.sliding_window_view(q_lagged, M)
        sigs  = np.dot(wins, w)
        signs = np.sign(sigs)
        signs[signs == 0] = 1.0
        out[M-1:] = signs
    return out

def compute_injection_signal(host, influencer, M_host, M_inf, w_inject, seed=0):
    rng      = np.random.RandomState(seed)
    inf_sig  = compute_signal(influencer, M_inf)
    inj_q    = q_lagged.copy()
    mask     = (rng.rand(n) < w_inject) & (~np.isnan(inf_sig))
    inj_q[mask] = inf_sig[mask]
    # rebuild host signal on injected queue
    w    = get_weights(host, M_host)
    out  = np.full(n, np.nan)
    if n >= M_host:
        wins  = np.lib.stride_tricks.sliding_window_view(inj_q, M_host)
        sigs  = np.dot(wins, w)
        signs = np.sign(sigs)
        signs[signs == 0] = 1.0
        out[M_host-1:] = signs
    return out

def metrics(pos_series, ret_series):
    """Annualised Sharpe and DAILY TURNOVER (% of portfolio repositioned per day)."""
    strat_ret = np.where(np.isnan(pos_series), 0.0, pos_series * ret_series)
    valid     = strat_ret[strat_ret != 0.0] if strat_ret.sum() != 0 else strat_ret
    sharpe    = 0.0
    if len(valid) > 1 and np.std(valid) > 0:
        sharpe = np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252)

    # Correct turnover: fraction of portfolio that changes position each day
    # Position is ±1 (long/short whole portfolio).
    # Change from +1→-1 = 2 units, 0→±1 = 1 unit.  Normalise to [0,1] per day.
    pos_clean = np.nan_to_num(pos_series, nan=0.0)
    pos_diff  = np.abs(np.diff(pos_clean)) / 2.0   # max change = 2 → normalise to 1
    turnover_pct = np.mean(pos_diff) * 100.0        # in percent
    return sharpe, turnover_pct

# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE sweep
# ─────────────────────────────────────────────────────────────────────────────
personalities = ['pushover', 'opportunist', 'traditionalist', 'contrarian']

PALETTE = {
    'pushover':       '#1a6faf',
    'opportunist':    '#e07b22',
    'traditionalist': '#2da44e',
    'contrarian':     '#c0392b',
    'curmudgeon':     '#8e44ad',
}
LABELS = {
    'pushover':       'Pushover (Flat)',
    'opportunist':    'Opportunist (Recency)',
    'traditionalist': 'Traditionalist (Distance)',
    'contrarian':     'Contrarian (Inverted)',
    'curmudgeon':     'Curmudgeon (MA-200 baseline)',
}

res_sa = {p: dict(is_sh=[], oos_sh=[], is_trn=[], oos_trn=[]) for p in personalities}

for p in personalities:
    for M in M_SWEEP:
        pos = compute_signal(p, M)
        sh_is,  trn_is  = metrics(pos[:split_idx],  ret[:split_idx])
        sh_oos, trn_oos = metrics(pos[split_idx:],  ret[split_idx:])
        res_sa[p]['is_sh'].append(sh_is)
        res_sa[p]['oos_sh'].append(sh_oos)
        res_sa[p]['is_trn'].append(trn_is)
        res_sa[p]['oos_trn'].append(trn_oos)

# Curmudgeon baseline (constant signal, no M-sweep)
curm_pos = compute_signal('curmudgeon', 1)
sh_is_c,  trn_is_c  = metrics(curm_pos[:split_idx], ret[:split_idx])
sh_oos_c, trn_oos_c = metrics(curm_pos[split_idx:], ret[split_idx:])

print(f"\n[DEBUG] Curmudgeon IS Sharpe={sh_is_c:.3f}, OOS={sh_oos_c:.3f}, IS Trn={trn_is_c:.2f}%, OOS Trn={trn_oos_c:.2f}%")
for p in personalities:
    print(f"[DEBUG] {p.upper()} IS Trn range: {min(res_sa[p]['is_trn']):.2f}%-{max(res_sa[p]['is_trn']):.2f}%  "
          f"OOS Trn range: {min(res_sa[p]['oos_trn']):.2f}%-{max(res_sa[p]['oos_trn']):.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Standalone
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220)
ax_sh, ax_trn = axes

for p in personalities:
    c = PALETTE[p]
    ax_sh.plot(M_SWEEP, res_sa[p]['oos_sh'],  color=c, lw=2.2, marker='o', ms=5, zorder=4)
    ax_sh.plot(M_SWEEP, res_sa[p]['is_sh'],   color=c, lw=1.4, marker='^', ms=4,
               linestyle='--', alpha=0.60, zorder=3)
    ax_trn.plot(M_SWEEP, res_sa[p]['oos_trn'], color=c, lw=2.2, marker='o', ms=5, zorder=4)
    ax_trn.plot(M_SWEEP, res_sa[p]['is_trn'],  color=c, lw=1.4, marker='^', ms=4,
                linestyle='--', alpha=0.60, zorder=3)

# Curmudgeon horizontal baselines
ax_sh.axhline(sh_oos_c, color=PALETTE['curmudgeon'], lw=2.0, ls='-',  zorder=5)
ax_sh.axhline(sh_is_c,  color=PALETTE['curmudgeon'], lw=1.4, ls='--', alpha=0.60, zorder=4)
ax_sh.axhline(0, color='k', ls=':', lw=1.0, alpha=0.5)
ax_trn.axhline(trn_oos_c, color=PALETTE['curmudgeon'], lw=2.0, ls='-',  zorder=5)
ax_trn.axhline(trn_is_c,  color=PALETTE['curmudgeon'], lw=1.4, ls='--', alpha=0.60, zorder=4)

# Compute data-driven y-limits for turnover
all_trn = ([trn_oos_c, trn_is_c] +
           [v for p in personalities for v in res_sa[p]['is_trn'] + res_sa[p]['oos_trn']])
trn_max = max(all_trn) * 1.15 if max(all_trn) > 0 else 5.0
trn_min = 0.0

ax_sh.set_title("(a) Annualised Sharpe Ratio vs. Memory Depth $M$",
                fontsize=12, fontweight='bold', pad=8)
ax_sh.set_xlabel("Cognitive Memory Depth $M$ (days)", fontsize=11)
ax_sh.set_ylabel("Annualised Sharpe Ratio", fontsize=11)
ax_sh.set_xticks(M_SWEEP)
ax_sh.tick_params(axis='x', labelsize=8)
ax_sh.grid(True, ls='--', alpha=0.5)

ax_trn.set_title("(b) Daily Portfolio Turnover (%) vs. Memory Depth $M$",
                 fontsize=12, fontweight='bold', pad=8)
ax_trn.set_xlabel("Cognitive Memory Depth $M$ (days)", fontsize=11)
ax_trn.set_ylabel("Daily Turnover (% portfolio repositioned)", fontsize=11)
ax_trn.set_xticks(M_SWEEP)
ax_trn.tick_params(axis='x', labelsize=8)
ax_trn.set_ylim(trn_min, trn_max)
ax_trn.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax_trn.grid(True, ls='--', alpha=0.5)

# Figure-level legend
handles = (
    [Line2D([0],[0], color=PALETTE[p], lw=2.2, marker='o', ms=5, label=LABELS[p])
     for p in personalities]
    + [Line2D([0],[0], color=PALETTE['curmudgeon'], lw=2.0, label=LABELS['curmudgeon'])]
    + [Line2D([0],[0], color='k', lw=2.0, ls='-',  marker='o', ms=5, label='Out-of-Sample (OOS 2021–25)'),
       Line2D([0],[0], color='k', lw=1.4, ls='--', marker='^', ms=4, alpha=0.7, label='In-Sample (IS 2015–20)')]
)
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.10),
           ncol=4, fontsize=9, frameon=True, framealpha=0.96, edgecolor='#cccccc')

fig.suptitle("Stage 1 – Standalone Kernels: IS (2015–2020) & OOS (2021–2025) "
             "Sharpe & Turnover Sweeps",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])

out1 = os.path.join(OUTPUT_DIR, "standalone_sharpe_turnover_vs_M.png")
fig.savefig(out1, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved standalone sweep → {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# COMBINATION ARCHITECTURES sweep
# ─────────────────────────────────────────────────────────────────────────────
combo_cfg = {
    'inj_opp_trad': {
        'label': 'Queue Inj: Opp(M) + Trad 5, w=0.50',
        'color': '#e07b22',
    },
    'inj_push_cont': {
        'label': 'Queue Inj: Push(M) + Cont(M), w=0.25',
        'color': '#2da44e',
    },
    'blend_opp_trad': {
        'label': 'Kernel Blend: Opp(M) + Trad 5, 50/50',
        'color': '#1a6faf',
    },
    'regime_switch': {
        'label': 'Regime Switch: Opp(M) ↔ Cont(M) [Vol-Gated]',
        'color': '#8e44ad',
    },
    'ensemble_vote': {
        'label': 'Ensemble Vote: Majority of 5 Kernels',
        'color': '#c0392b',
    },
}

res_cb = {k: dict(is_sh=[], oos_sh=[], is_trn=[], oos_trn=[]) for k in combo_cfg}

for M in M_SWEEP:
    sig_opp  = compute_signal('opportunist',    M)
    sig_trad = compute_signal('traditionalist', 5)
    sig_push = compute_signal('pushover',       M)
    sig_cont = compute_signal('contrarian',     M)
    sig_trad_M = compute_signal('traditionalist', M)

    # 1. Opp host + Trad influencer injection
    pos1 = compute_injection_signal('opportunist', 'traditionalist', M, 5, 0.50)
    # 2. Push host + Cont influencer injection
    pos2 = compute_injection_signal('pushover', 'contrarian', M, M, 0.25)
    # 3. Kernel blend
    raw3  = 0.5 * sig_opp + 0.5 * sig_trad
    pos3  = np.sign(raw3); pos3[pos3 == 0] = 1.0
    # 4. Regime switch
    pos4  = np.where(vol21 <= vol_thresh, sig_opp, sig_cont)
    pos4  = np.sign(pos4); pos4[pos4 == 0] = 1.0
    # 5. Ensemble vote
    raw5  = sig_push + sig_opp + sig_trad_M + sig_cont + curm_sig
    pos5  = np.sign(raw5); pos5[pos5 == 0] = 1.0

    for key, pos in zip(list(combo_cfg.keys()), [pos1, pos2, pos3, pos4, pos5]):
        sh_is,  trn_is  = metrics(pos[:split_idx], ret[:split_idx])
        sh_oos, trn_oos = metrics(pos[split_idx:], ret[split_idx:])
        res_cb[key]['is_sh'].append(sh_is)
        res_cb[key]['oos_sh'].append(sh_oos)
        res_cb[key]['is_trn'].append(trn_is)
        res_cb[key]['oos_trn'].append(trn_oos)

for k in combo_cfg:
    print(f"[DEBUG] {k} IS Trn range: {min(res_cb[k]['is_trn']):.2f}%-{max(res_cb[k]['is_trn']):.2f}%  "
          f"OOS Trn range: {min(res_cb[k]['oos_trn']):.2f}%-{max(res_cb[k]['oos_trn']):.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Combination
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220)
ax_sh2, ax_trn2 = axes

for key, cfg in combo_cfg.items():
    c = cfg['color']
    ax_sh2.plot(M_SWEEP, res_cb[key]['oos_sh'],  color=c, lw=2.2, marker='o', ms=5, zorder=4)
    ax_sh2.plot(M_SWEEP, res_cb[key]['is_sh'],   color=c, lw=1.4, marker='^', ms=4,
                linestyle='--', alpha=0.60, zorder=3)
    ax_trn2.plot(M_SWEEP, res_cb[key]['oos_trn'], color=c, lw=2.2, marker='o', ms=5, zorder=4)
    ax_trn2.plot(M_SWEEP, res_cb[key]['is_trn'],  color=c, lw=1.4, marker='^', ms=4,
                 linestyle='--', alpha=0.60, zorder=3)

ax_sh2.axhline(0, color='k', ls=':', lw=1.0, alpha=0.5)

all_cb_trn = [v for key in combo_cfg for v in res_cb[key]['is_trn'] + res_cb[key]['oos_trn']]
cb_trn_max = max(all_cb_trn) * 1.15 if max(all_cb_trn) > 0 else 5.0

ax_sh2.set_title("(a) Annualised Sharpe Ratio vs. Host Memory Depth $M$",
                 fontsize=12, fontweight='bold', pad=8)
ax_sh2.set_xlabel("Host Memory Depth $M$ (days)", fontsize=11)
ax_sh2.set_ylabel("Annualised Sharpe Ratio", fontsize=11)
ax_sh2.set_xticks(M_SWEEP)
ax_sh2.tick_params(axis='x', labelsize=8)
ax_sh2.grid(True, ls='--', alpha=0.5)

ax_trn2.set_title("(b) Daily Portfolio Turnover (%) vs. Host Memory Depth $M$",
                  fontsize=12, fontweight='bold', pad=8)
ax_trn2.set_xlabel("Host Memory Depth $M$ (days)", fontsize=11)
ax_trn2.set_ylabel("Daily Turnover (% portfolio repositioned)", fontsize=11)
ax_trn2.set_xticks(M_SWEEP)
ax_trn2.tick_params(axis='x', labelsize=8)
ax_trn2.set_ylim(0, cb_trn_max)
ax_trn2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax_trn2.grid(True, ls='--', alpha=0.5)

handles2 = (
    [Line2D([0],[0], color=cfg['color'], lw=2.2, marker='o', ms=5, label=cfg['label'])
     for cfg in combo_cfg.values()]
    + [Line2D([0],[0], color='k', lw=2.0, ls='-',  marker='o', ms=5, label='Out-of-Sample (OOS 2021–25)'),
       Line2D([0],[0], color='k', lw=1.4, ls='--', marker='^', ms=4, alpha=0.7, label='In-Sample (IS 2015–20)')]
)
fig.legend(handles=handles2, loc='lower center', bbox_to_anchor=(0.5, -0.12),
           ncol=3, fontsize=9, frameon=True, framealpha=0.96, edgecolor='#cccccc')

fig.suptitle("Stage 1 – Combination Architectures: IS (2015–2020) & OOS (2021–2025) "
             "Sharpe & Turnover Sweeps",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 1])

out2 = os.path.join(OUTPUT_DIR, "combination_sharpe_turnover_vs_M.png")
fig.savefig(out2, dpi=220, bbox_inches='tight')
plt.close(fig)
print(f"Saved combination sweep → {out2}")
print("\n✅ Done.")
