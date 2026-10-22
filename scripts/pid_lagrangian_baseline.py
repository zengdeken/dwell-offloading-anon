"""PID-Lagrangian adaptive-penalty baseline (Stooke et al., ICML 2020).

Identical training protocol to the joint fixed-penalty baseline
(per-contract long blocks, shuffled cycles, budget-symmetric), except the
penalty multiplier is NOT fixed: after every PPO update the multiplier is
moved by a PID controller acting on the measured positive-part violation

    lam <- clip(lam + kp*J + ki*sum(J) + kd*(J - J_prev), [lam_min, lam_max])

where J is the mean episode violation since the last update. This baseline
answers the reviewer question the letter argued but did not measure: can an
adaptive penalty substitute for dwell-aware training at the same protocol?
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import Params
from src.env import QoSOffloadEnv
from src.eval_metrics import CONTRACTS, contract_metrics
from src.ppo import PPOAgent
from src.runner import _save_history, evaluate, make_policy, rollout


def atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def apply_contract(env, name):
    spec = CONTRACTS[name]
    env.set_contract(spec["eta_min"], spec["t_max"])


def train_pid(cfg, episodes_per_contract, contract_names, block_episodes,
              pid, resume=False):
    total_episodes = episodes_per_contract * len(contract_names)
    cfg.episodes = total_episodes
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = QoSOffloadEnv(cfg)
    agent = PPOAgent(cfg)
    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    atomic_json(os.path.join(run_dir, "pid_protocol.json"), {
        "method": "PID-Lagrangian adaptive penalty",
        "contracts": {name: CONTRACTS[name] for name in contract_names},
        "episodes_per_contract": episodes_per_contract,
        "contract_block_episodes": block_episodes,
        "total_episodes": total_episodes,
        "lambda0": pid["lambda0"], "kp": pid["kp"], "ki": pid["ki"],
        "kd": pid["kd"], "lambda_min": pid["lambda_min"],
        "lambda_max": pid["lambda_max"],
        "schedule": "deterministic shuffled long-contract-block cycles",
    })

    buf = {"obs": [], "acts": [], "logp": [], "rew": [],
           "val": [], "dones": []}
    history = []
    viol_window = []          # episodes since last update
    lam = pid["lambda0"]
    J_prev, J_sum = None, 0.0
    cfg.fixed_penalty = lam
    last_stats = {}

    def flush(bootstrap):
        values = buf["val"] + [bootstrap]
        stats = agent.update(buf["obs"], buf["acts"], buf["logp"],
                             buf["rew"], buf["dones"], values)
        for key in buf:
            buf[key] = []
        return stats

    for ep in range(1, total_episodes + 1):
        names = sorted(contract_names)
        block_index, within = divmod(ep - 1, block_episodes)
        cycle, offset = divmod(block_index, len(names))
        rng = np.random.default_rng(cfg.seed * 1000003 + cycle)
        cname = list(rng.permutation(names))[offset]
        apply_contract(env, cname)
        env.rng = np.random.default_rng(cfg.seed * 100000 + ep)
        if within == 0:
            env.reset()
        policy = make_policy(cfg, env, agent)
        traj, logs = rollout(cfg, env, policy, agent=agent, reset=False)
        metrics = env.episode_metrics(logs)

        buf["obs"].extend(traj["obs"])
        buf["acts"].extend(traj["acts"])
        buf["logp"].extend(traj["logp"])
        buf["rew"].extend(r / cfg.reward_scale for r in traj["rew"])
        buf["val"].extend(traj["val"][:-1])
        block_dones = [False] * len(traj["rew"])
        if within == block_episodes - 1 and ep < total_episodes:
            block_dones[-1] = True
        buf["dones"].extend(block_dones)
        last_bootstrap = traj["val"][-1]

        update_applied = False
        if ep % cfg.update_every == 0:
            last_stats = flush(last_bootstrap)
            update_applied = True

        spec = CONTRACTS[cname]
        compliance = contract_metrics(
            metrics["tier_completion"], metrics["tier_norm_delay"],
            spec["eta_min"], spec["t_max"], None)
        entry = {
            "episode": ep,
            "contract": cname,
            "lambda_f": float(lam),
            "reward": float(np.sum(traj["rew"])),
            "cost": metrics["cost"],
            "violation": metrics["violation"],
            "tier_completion": metrics["tier_completion"].tolist(),
            "tier_norm_delay": metrics["tier_norm_delay"].tolist(),
            "completion_met": compliance["completion_met"],
            "delay_met": compliance["delay_met"],
            "full_contract_met": compliance["full_contract_met"],
            "eta_margin": compliance["eta_margin"],
            "delay_margin": compliance["delay_margin"],
            "Z_final": metrics["Z_final"].tolist(),
            "Y_final": metrics["Y_final"].tolist(),
            "update_applied": update_applied,
        }
        if update_applied:
            entry["update_index"] = ep
            entry["rollout_samples"] = last_stats["rollout_samples"]
            # PID step on the windowed violation
            J = float(np.mean([h["violation"] for h in viol_window]))
            dJ = 0.0 if J_prev is None else J - J_prev
            lam = max(pid["lambda_min"],
                      min(pid["lambda_max"],
                          lam + pid["kp"] * J + pid["ki"] * J_sum + pid["kd"] * dJ))
            cfg.fixed_penalty = lam          # propagates to env reward live
            entry["pid_J"] = J
            entry["lambda_after_pid"] = float(lam)
            J_prev, J_sum = J, J_sum + J
            viol_window = []
        history.append(entry)
        viol_window.append(entry)

        if ep % 50 == 0:
            _save_history(history, run_dir)
        if ep % max(1, total_episodes // 10) == 0:
            print(f"[pid {cfg.run_name}] {ep}/{total_episodes} {cname} "
                  f"lam={lam:.1f} cost={metrics['cost']:.3f} "
                  f"viol={metrics['violation']:.3f}", flush=True)

    if buf["rew"]:
        last_stats = flush(last_bootstrap)
        history[-1]["update_applied"] = True
        history[-1]["rollout_samples"] = last_stats["rollout_samples"]

    agent.save(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
    atomic_json(os.path.join(run_dir, "pid_trajectory.json"), {
        "lambda0": pid["lambda0"], "kp": pid["kp"], "ki": pid["ki"],
        "kd": pid["kd"], "final_lambda": float(lam),
        "trajectory": [
            {"episode": h["episode"], "lambda_f": h["lambda_f"],
             "lambda_after_pid": h.get("lambda_after_pid"),
             "pid_J": h.get("pid_J")}
            for h in history if h["update_applied"]],
    })
    _save_history(history, run_dir)
    return os.path.join(run_dir, "agent.pt")


def evaluate_contracts(cfg, weights_path, contract_names):
    rows = []
    parent_run = cfg.run_name
    for cname in contract_names:
        spec = CONTRACTS[cname]
        ecfg = copy.deepcopy(cfg)
        ecfg.eta_min = list(spec["eta_min"])
        ecfg.t_max = list(spec["t_max"])
        ecfg.run_name = os.path.join(parent_run, "eval", cname)
        ecfg = ecfg.finalize()
        summary = evaluate(ecfg, weights_from=weights_path)
        compliance = contract_metrics(
            summary["tier_completion"], summary["tier_norm_delay"],
            spec["eta_min"], spec["t_max"],
            summary.get("contract_stationary"))
        rows.append({
            "contract": cname,
            "lambda_f": cfg.fixed_penalty,
            "seed": cfg.seed,
            "violation": summary["violation"],
            "cost": summary["cost"],
            "tier_completion": summary["tier_completion"],
            "tier_norm_delay": summary["tier_norm_delay"],
            "physical_stationary": summary.get("physical_stationary"),
            "price_stationary": summary.get("price_stationary"),
            "contract_stationary": summary.get("contract_stationary"),
            "method": "pid_lagrangian",
            **compliance,
        })
    out = os.path.join(cfg.out_root, parent_run, "joint_eval.json")
    atomic_json(out, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda0", type=float, default=50.0)
    ap.add_argument("--kp", type=float, default=25.0)
    ap.add_argument("--ki", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=5.0)
    ap.add_argument("--lambda_min", type=float, default=0.0)
    ap.add_argument("--lambda_max", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes_per_contract", type=int, default=1000)
    ap.add_argument("--time_slots", type=int, default=100)
    ap.add_argument("--update_every", type=int, default=2)
    ap.add_argument("--contract_block_episodes", type=int, default=1)
    ap.add_argument("--out_root", default="results/pid_lagrangian")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    contract_names = sorted(CONTRACTS)
    cfg = Params().finalize()
    cfg.reward_mode = "fixed"
    cfg.aging_mode = "constant"
    cfg.seed = args.seed
    cfg.eval_seed = 2345
    cfg.time_slots = args.time_slots
    cfg.update_every = args.update_every
    cfg.out_root = args.out_root
    episodes_per_contract = args.episodes_per_contract
    block_episodes = args.contract_block_episodes
    if args.smoke:
        episodes_per_contract = 1
        block_episodes = 1
    pid = {"lambda0": args.lambda0, "kp": args.kp, "ki": args.ki,
           "kd": args.kd, "lambda_min": args.lambda_min,
           "lambda_max": args.lambda_max}
    cfg.fixed_penalty = args.lambda0
    cfg.run_name = (f"pid{args.lambda0:g}_kp{args.kp:g}_ki{args.ki:g}"
                    f"_kd{args.kd:g}_b{block_episodes}_s{args.seed}"
                    f"{('_' + args.tag) if args.tag else ''}")
    if args.smoke:
        cfg.time_slots = 8
        cfg.eval_episodes = 2
        cfg.min_burn_slots = 20
        cfg.burn_window = 10
        cfg.max_burn_slots = 30
        cfg.checkpoint_every = 0
    cfg = cfg.finalize()

    weights = os.path.join(cfg.out_root, cfg.run_name, "agent.pt")
    if args.stage in ("all", "train"):
        weights = train_pid(cfg, episodes_per_contract, contract_names,
                            block_episodes, pid)
    if args.stage in ("all", "eval"):
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        rows = evaluate_contracts(cfg, weights, contract_names)
        viol = [r["violation"] for r in rows]
        print(f"[pid summary] mean violation {np.mean(viol):.4f} "
              f"over {len(rows)} contracts")


if __name__ == "__main__":
    main()
