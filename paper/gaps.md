# Paper Audit & Gaps Analysis: Physics-Informed Market Strategy (PIMS)

**Date**: August 27, 2026  
**Status**: Complete, Audited, Publication-Ready  
**Master LaTeX Document**: `paper/main.tex`

---

## 1. Quantitative Verification Status
- **Unverified Claims**: **0** (All quantitative claims trace directly to verified data records and simulation scripts).
- **Remaining `\todo` Tags**: **0**.
- **Prose Integrity**: Strictly verified for zero em dashes (`--` or `---` in prose replaced with parentheses, colons, or commas).

---

## 2. Quantitative Claims & Verified Traceability Mapping

| # | Scientific Claim | Exact Number / Supporting Metric | Verified Source Script / Asset |
| :-: | :--- | :--- | :--- |
| 1 | **Top Out-of-Sample Return Combination** | $\text{Sharpe}_{\text{OOS}} = 1.486$, $\text{Ret}_{\text{ann}} = +18.4\%$, $\text{MaxDD} = -14.6\%$, Turnover $= 44.3\%$ | `src/generate_combination_is_oos_plots.py` |
| 2 | **Optimal Turnover-Adjusted Combination** | $\text{Sharpe}_{\text{OOS}} = 1.470$, $\text{Ret}_{\text{ann}} = +16.8\%$, $\text{MaxDD} = -16.4\%$, Turnover $= 15.2\%$ | `src/generate_combination_is_oos_plots.py` |
| 3 | **Subordinate Combination Performance** | Kernel Blending $\text{Sharpe}_{\text{OOS}} = 0.627$; Regime Switching $\text{Sharpe}_{\text{OOS}} = 0.331$; Ensemble Voting $\text{Sharpe}_{\text{OOS}} = 0.153$; Pushover Baseline $\text{Sharpe}_{\text{OOS}} = 0.957$; Buy & Hold $\text{Sharpe}_{\text{OOS}} = 0.820$ | `src/generate_combination_is_oos_plots.py` |
| 4 | **Top 5 Stable Standalone Kernels ($M \in [1, 21]$)** | 1. Pushover $M=9$ ($\text{IS}=0.85, \text{OOS}=0.96, \text{Gap}=0.11, \text{Ratio}=1.13$)<br>2. Pushover $M=8$ ($\text{IS}=0.74, \text{OOS}=1.05, \text{Gap}=0.31, \text{Ratio}=1.42$)<br>3. Pushover $M=10$ ($\text{IS}=0.73, \text{OOS}=0.64, \text{Gap}=0.09, \text{Ratio}=0.88$)<br>4. Traditionalist $M=19$ ($\text{IS}=0.51, \text{OOS}=0.63, \text{Gap}=0.12, \text{Ratio}=1.24$)<br>5. Opportunist $M=9$ ($\text{IS}=0.44, \text{OOS}=0.57, \text{Gap}=0.13, \text{Ratio}=1.28$) | `src/analyze_top5_stable_standalone.py` |
| 5 | **Theoretical Critical Memory Threshold** | $M_c = \frac{\pi}{2(1-2\eta)^2} \approx 9.82$ trading days (for cognitive noise $\eta = 0.30$) | `src/thalf_experiment.py`, `paper/stage2_swarm_physics_paper.tex` |
| 6 | **Empirical Pitchfork Bifurcation Calibration** | Potential curvature $k(M)$ crosses zero between $M=9$ and $M=11$ for peer crowds ($p_{\text{peer}}=1.0$) | `src/mixed_sim_both_crowds.py` |
| 7 | **Impossibility of Herding for $p_{\text{peer}}=0$** | Curvature $k > 0$ strictly across all $M \in [1, 21]$ and $\eta \in [0, 0.45]$ for market-only conformists | `src/heatmap_market_only.py` |
| 8 | **Panic Relaxation Critical Slowing Down** | Pushover $T_{1/2}(M=1)=3\text{d} \to T_{1/2}(M=5)=33\text{d} \to T_{1/2}(M=11)=1919\text{d} \to T_{1/2}(M \ge 13)=\text{DNF} / \infty$ | `src/thalf_experiment.py` |
| 9 | **Curmudgeon Circuit Breaking** | Introducing $\phi_C \ge 10\%$ structural anchors caps recovery at $T_{1/2} \le 30\text{--}60$ days across all $M$ | `src/thalf_experiment.py` |
| 10 | **Physics-Informed PIMS Strategy Benchmark** | Naive In-Sample Selection ($\text{IS}=1.001 \to \text{OOS}=0.399, \text{Gap}=0.602$)<br>PIMS Physics Filter ($M_{\text{eff}} \le M_c$): ($\text{IS}=0.938 \to \text{OOS}=0.564, \text{Gap}=0.375$)<br>Gain: **$+41.4\%$ Out-of-Sample Sharpe**, **$37.7\%$ Overfitting Gap Reduction** | `src/analyze_unified_pims_connection.py` |
| 11 | **Potential Barrier Parameter Prediction** | Theoretical optimal memory $M^*_{\text{predicted}} = \lfloor M_c \rfloor = 9$ days; Potential barrier $\Delta V(M) = \frac{k(M)^2}{4 c(M)}$ correlates linearly with overfitting gap $\Delta \text{Sharpe}$ | `src/analyze_potential_parameter_prediction.py` |

---

## 3. Explicit Record of Judgment Calls

1. **Cognitive Noise Parameter Calibration ($\eta = 0.30$)**:
   - *Rationale*: Corresponds to a 30% misinterpretation error rate in human news / sentiment transmission over social networks. Sensitivity sweeps across $\eta \in [0.10, 0.45]$ confirm that the qualitative pitchfork bifurcation is invariant.
2. **In-Sample / Out-of-Sample Split Boundary (60/40 Split)**:
   - *Rationale*: 60% IS (1,485 days: 2015–2020) encapsulates both quiet bull regimes and high-volatility stress periods (2016 demonetization and 2020 COVID crash), while 40% OOS (991 days: 2021–2025) provides a clean post-pandemic validation sample.
3. **Macro Trend Anchor Definition (Curmudgeon)**:
   - *Rationale*: Standard 200-day simple moving average slope ($\text{sign}(\Delta \text{MA}_{200})$), representing classic institutional trend-following capital.
4. **Subgrid Taxonomy**:
   - 784 configurations = Core 4-host $\times$ 4-influencer $\times$ 4-$M$ $\times$ 4-$M$ $\times$ 3-$w_{\text{inject}}$ interaction grid.
   - 797 configurations = Core interaction grid + standalone and equal-mixture baselines.
   - 852 configurations = Complete spectrum expanding standalone sweeps across $M \in [1, 21]$.
