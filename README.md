# Dwell-Time-Aware Training for Differentiated-QoS Edge Offloading

Code and frozen evaluation records for the paper "One Penalty, Multiple
Contracts: Dwell-Time-Aware Training for Differentiated-QoS Edge
Offloading" (double-blind review copy).

Contract-conditioned PPO for MEC offloading in which the training protocol
organizes episodes into dwell blocks of $B$ episodes per QoS contract: state
persists within a block, and state resets plus GAE truncation happen only at
block boundaries.

## Layout

```
src/
  config.py        all parameters (dataclass; finalize() derives the rest)
  env.py           three-stage MEC environment (queues, dual edge pools,
                   virtual debt queues, contracts, observation)
  ppo.py           zero-inflated Beta actor + independent critic (PPO)
  runner.py        rollout / evaluation loop
  eval_metrics.py  contract compliance (FullStrict) accounting
  baselines.py     greedy / DPP reference controllers
  dpp_controller.py
scripts/
  joint_fixed_baseline.py   main protocol (dwell-aware block training);
                            --contract_block_episodes B, --gae_cut_each_episode
                            for the credit-horizon control
  pid_lagrangian_baseline.py adaptive-penalty baseline (PID-Lagrangian)
  gradsurgery_baseline.py    gradient-surgery baseline (PCGrad)
results_summary/   frozen CSVs: per-block sweep, method comparison,
                   generalization, expansion runs, T_rec measurements
eval_records/      per-run joint_eval.json for the rev2 controls
                   (epcut, B=3/B=4 boundary, B x H crossing, PID gains)
```

## Quick start

```
pip install torch numpy
# smoke test (a few minutes on CPU):
python scripts/joint_fixed_baseline.py --lambda_f 50 --seed 1234 \
    --episodes_per_contract 2 --contract_block_episodes 2 \
    --time_slots 8 --smoke --out_root results/smoke
# main protocol (B=50, full training):
python scripts/joint_fixed_baseline.py --lambda_f 50 --seed 1234 \
    --contract_block_episodes 50 --out_root results/joint_fixed
```

Each training run writes `history.json`, `joint_protocol.json` (exact
schedule), and `config.json` into its run directory, then evaluates on all
training contracts into `joint_eval.json`.

## Where the paper numbers come from

| Paper artifact | Source in this repo |
|---|---|
| Table "dwell sweep" (B in {1,2,...,100}) | `results_summary/block_sweep_summary.csv` |
| Method comparison (Table 3) | `results_summary/main_comparison.csv`, PID/PCGrad rows reconstructable from `results_summary/expansion_runs_summary.csv` |
| Controls (epcut / B x H crossing) | `eval_records/*epcut*`, `eval_records/*cross*` |
| Generalization (K4/K5) | `results_summary/generalization_summary.csv` |
| Debt-excursion measurements | `results_summary/trec_measurements.json` |

Statistics convention: mean +/- population std over seed-level means.
