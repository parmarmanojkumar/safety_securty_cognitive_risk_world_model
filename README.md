# Supplemental Material — CCS 2026 Submission

## Safety, Security, and Cognitive Risks in World Models

**Anonymised for double-blind review.**
This archive contains all code and pre-computed results needed to reproduce
the empirical claims in the paper (Section 6 and Appendix E).

---

## Contents

```
ccs_supplemental/
├── README.md                          ← this file
├── code/
│   ├── run_world_model_experiments.py ← Main experiment (GRU-RSSM, RSSM proxy,
│   │                                    PGD-10 mitigation, reward gap)
│   └── run_dreamerv3_checkpoint_attack.py ← DreamerV3 checkpoint probe
├── results/
│   ├── summary_metrics.json           ← All key paper metrics in one place
│   ├── tables/
│   │   ├── gru_wm_trajectory_metrics.csv           ← Panel A data (GRU before mitigation)
│   │   ├── gru_wm_robust_trajectory_metrics.csv    ← Panel D data (GRU after PGD-10)
│   │   ├── rssm_proxy_wm_trajectory_metrics.csv    ← Panel B data (RSSM proxy)
│   │   ├── mitigation_reduction_summary.csv        ← A_k reduction percentages
│   │   ├── reward_gap_with_baseline.csv            ← Panel F data (reward gap)
│   │   ├── budget_curve_before_mitigation.csv      ← Panel E data (budget, before)
│   │   ├── budget_curve_after_mitigation.csv       ← Panel E data (budget, after)
│   │   ├── dreamerv3_checkpoint_patch_summary.csv  ← Panel C data (DreamerV3 probe)
│   │   ├── dreamerv3_checkpoint_patch_budget.csv   ← DreamerV3 patch-size ablation
│   │   └── dreamerv3_checkpoint_vector_nullcheck.csv ← DreamerV3 vector null check
│   └── figures/
│       ├── paper_figure.pdf           ← Figure 1 as used in the paper (Panels A–F)
│       └── paper_figure.png           ← Same figure, PNG format
```

---

## Quick Verification of Key Paper Numbers

All metrics below are in `results/summary_metrics.json`:

| Paper claim                              | JSON key                         | Value              |
| ---------------------------------------- | -------------------------------- | ------------------ |
| GRU amplification A1 (before mitigation) | `A1_gru_asym_before`             | **2.26×**          |
| GRU amplification A1 (after PGD-10)      | `A1_gru_asym_after`              | **0.92×** (−59.5%) |
| Stochastic RSSM proxy A1                 | `A1_rssm_asym`                   | **0.65×**          |
| WM reward gap at H=30                    | `wm_gap_H30`                     | **0.000892**       |
| SS reward gap at H=30                    | `ss_gap_H30`                     | **0.000175**       |
| DreamerV3 A1 (image patch)               | `A1_dreamerv3_patch_mean`        | **0.0262**         |
| DreamerV3 latent error E1                | `E1_dreamerv3_patch_mean`        | **1.4523**         |
| DreamerV3 action drift ‖Δa1‖             | `E_action1_dreamerv3_patch_mean` | **0.0080**         |

---

## Reproducing the GRU/RSSM Experiments (Experiment 1)

### Requirements

```
Python >= 3.9
torch >= 2.0
numpy, pandas, matplotlib
```

Install with:

```bash
pip install torch numpy pandas matplotlib
```

### Run

```bash
cd code/
python run_world_model_experiments.py
```

By default, outputs are written to `./outputs/` (created automatically).
All random seeds are fixed (`seed=42`); results match `results/summary_metrics.json`
within Monte Carlo variance across `N=200` trials.

**Expected runtime**: ~3–8 minutes on a modern CPU (no GPU required).

**Key output files produced**:

- `outputs/tables/gru_wm_trajectory_metrics.csv` — matches `results/tables/gru_wm_trajectory_metrics.csv`
- `outputs/tables/mitigation_reduction_summary.csv` — matches paper Table values
- `outputs/figures/` — reproduces Figure 1 Panels A–F

---

## Reproducing the DreamerV3 Checkpoint Probe (Experiment 2)

This experiment requires a DreamerV3 installation and a trained checkpoint.

### Requirements

```
DreamerV3 repository: https://github.com/danijar/dreamerv3
  (public, no authentication required)
A DreamerV3 checkpoint — the 'dummy_disc' debug checkpoint used in the paper
  can be generated via: python dreamerv3/main.py --logdir /tmp/dreamerv3_run4
  (runs for ~5 min to produce a usable debug checkpoint)
ruamel.yaml, jax, flax (DreamerV3 dependencies)
```

### Run

```bash
cd code/
python run_dreamerv3_checkpoint_attack.py \
  --dreamer-repo /path/to/dreamerv3 \
  --run-dir /tmp/dreamerv3_run4 \
  --out-dir /tmp/dreamerv3_results \
  --horizon 30 \
  --attack-step 1 \
  --patch-size 32 \
  --patch-stride 16
```

Pre-computed results from this experiment are in
`results/tables/dreamerv3_checkpoint_patch_summary.csv` and
`results/summary_metrics.json` (`A1_dreamerv3_*` keys).

---

## Notes on Anonymisation

- Author names, institutional affiliations, and identifying file paths have been
  removed from all code files.
- The DreamerV3 repository referenced above (`danijar/dreamerv3`) is a public
  third-party repository and does not identify the authors of this submission.
- The experiment code will be released under an open-source licence upon acceptance.
