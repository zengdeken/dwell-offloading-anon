"""Fair K0-K3 joint fixed-penalty baseline.

Each contract receives the same number of independent 100-slot training
blocks. Physical queues and QoS debts reset at every block boundary, and GAE
is explicitly cut there so returns never cross contracts.
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


DIAG_KEYS = (
    "approx_kl", "clipfrac", "entropy", "value_error",
    "normalized_value_mae", "explained_variance", "critic_clipfrac",
    "mean_abs_return", "actor_grad_norm", "critic_grad_norm",
    "boundary_frac", "reward_scale",
)


def atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def contract_for_episode(episode, seed, contract_names=None):
    """Deterministic shuffled four-contract cycle, stable across resume."""
    return contract_block_for_episode(
        episode, seed, contract_names, block_episodes=1)[0]


def contract_block_for_episode(episode, seed, contract_names=None,
                               block_episodes=1):
    """Return contract and reset/cut flags for a long contract block."""
    names = sorted(contract_names or CONTRACTS)
    block_index, within = divmod(episode - 1, block_episodes)
    cycle, offset = divmod(block_index, len(names))
    rng = np.random.default_rng(seed * 1000003 + cycle)
    name = list(rng.permutation(names))[offset]
    return name, within == 0, within == block_episodes - 1


def apply_contract(env, name):
    spec = CONTRACTS[name]
    env.set_contract(spec["eta_min"], spec["t_max"])


def train_joint(cfg, episodes_per_contract, contract_names, block_episodes,
                state_bank=False, resume=False, gae_cut_each_episode=False):
    if episodes_per_contract % block_episodes != 0:
        raise ValueError("episodes_per_contract must be divisible by block_episodes")
    total_episodes = episodes_per_contract * len(contract_names)
    cfg.episodes = total_episodes
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = QoSOffloadEnv(cfg)
    agent = PPOAgent(cfg)
    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    atomic_json(os.path.join(run_dir, "joint_protocol.json"), {
        "contracts": {name: CONTRACTS[name] for name in contract_names},
        "episodes_per_contract": episodes_per_contract,
        "contract_block_episodes": block_episodes,
        "total_episodes": total_episodes,
        "reset_each_contract_block": True,
        "persistent_contract_state_bank": bool(state_bank),
        "gae_cut_each_contract_block": True,
        "gae_cut_each_episode": bool(gae_cut_each_episode),
        "schedule": "deterministic shuffled long-contract-block cycles",
    })

    start_ep = 1
    history = []
    contract_states = {}
    state_path = os.path.join(run_dir, "train_state.pt")
    if resume and os.path.exists(state_path):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        agent.load(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        if "env_state" not in ck:
            raise ValueError("checkpoint lacks env_state for long-block resume")
        env.load_state_dict(ck["env_state"])
        contract_states = ck.get("contract_states", {})
        start_ep = int(ck["episode"]) + 1
        hist_path = os.path.join(run_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
        print(f"[joint {cfg.run_name}] resume {start_ep - 1} -> "
              f"{total_episodes}", flush=True)

    buf = {"obs": [], "acts": [], "logp": [], "rew": [],
           "val": [], "dones": []}
    last_stats = {}

    def flush(bootstrap):
        values = buf["val"] + [bootstrap]
        stats = agent.update(buf["obs"], buf["acts"], buf["logp"],
                             buf["rew"], buf["dones"], values)
        for key in buf:
            buf[key] = []
        return stats

    last_bootstrap = 0.0
    for ep in range(start_ep, total_episodes + 1):
        cname, block_start, block_end = contract_block_for_episode(
            ep, cfg.seed, contract_names, block_episodes)
        apply_contract(env, cname)
        stream_seed = cfg.seed * 100000 + ep
        if block_start:
            if state_bank and cname in contract_states:
                env.load_state_dict(contract_states[cname])
                env.rng = np.random.default_rng(stream_seed)
            else:
                env.rng = np.random.default_rng(stream_seed)
                env.reset()
        else:
            env.rng = np.random.default_rng(stream_seed)
        policy = make_policy(cfg, env, agent)
        traj, logs = rollout(cfg, env, policy, agent=agent, reset=False)
        metrics = env.episode_metrics(logs)
        if state_bank and block_end:
            contract_states[cname] = env.state_dict()

        buf["obs"].extend(traj["obs"])
        buf["acts"].extend(traj["acts"])
        buf["logp"].extend(traj["logp"])
        buf["rew"].extend(r / cfg.reward_scale for r in traj["rew"])
        buf["val"].extend(traj["val"][:-1])
        block_dones = [False] * len(traj["rew"])
        # GAE cut at the block end; with gae_cut_each_episode it is cut at
        # EVERY episode end instead, while state still persists across
        # episodes within the block (credit-horizon control experiment).
        if (block_end or gae_cut_each_episode) and ep < total_episodes:
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
            for key in DIAG_KEYS:
                entry[key] = last_stats[key]
        history.append(entry)

        if (cfg.checkpoint_every and ep % cfg.checkpoint_every == 0
                and ep % cfg.update_every == 0 and not buf["rew"]):
            agent.save(os.path.join(run_dir, "agent.pt"),
                       with_optimizers=True)
            torch.save({
                "episode": ep,
                "env_state": env.state_dict(),
                "contract_states": contract_states,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }, state_path)
            _save_history(history, run_dir)

        if ep % max(1, total_episodes // 10) == 0:
            extra = (f" kl={last_stats['approx_kl']:.4f}"
                     if update_applied else "")
            print(f"[joint {cfg.run_name}] {ep}/{total_episodes} "
                  f"{cname} cost={metrics['cost']:.3f} "
                  f"viol={metrics['violation']:.3f}{extra}", flush=True)

    if buf["rew"]:
        last_stats = flush(last_bootstrap)
        history[-1]["update_applied"] = True
        history[-1]["rollout_samples"] = last_stats["rollout_samples"]
        for key in DIAG_KEYS:
            history[-1][key] = last_stats[key]

    agent.save(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
    torch.save({
        "episode": total_episodes,
        "env_state": env.state_dict(),
        "contract_states": contract_states,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }, state_path)
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
            **compliance,
        })
    out = os.path.join(cfg.out_root, parent_run, "joint_eval.json")
    atomic_json(out, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_f", type=float, choices=[50.0, 100.0],
                    required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes_per_contract", type=int, default=1000)
    ap.add_argument("--time_slots", type=int, default=100)
    ap.add_argument("--update_every", type=int, default=2)
    ap.add_argument("--contract_block_episodes", type=int, default=1)
    ap.add_argument("--gae_cut_each_episode", action="store_true",
                    help="cut the GAE recursion at every episode end while "
                         "state persists within the block (credit-horizon "
                         "control)")
    ap.add_argument("--state_bank", action="store_true")
    ap.add_argument("--out_root", default="results/joint_fixed")
    ap.add_argument(
        "--contracts", default=",".join(sorted(CONTRACTS)),
        help="comma-separated training/evaluation contracts",
    )
    ap.add_argument("--stage", choices=["all", "train", "eval"],
                    default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    contract_names = [x.strip() for x in args.contracts.split(",")
                      if x.strip()]
    unknown = sorted(set(contract_names) - set(CONTRACTS))
    if not contract_names or unknown:
        raise ValueError(f"invalid contracts: {unknown or contract_names}")

    cfg = Params().finalize()
    cfg.reward_mode = "fixed"
    cfg.aging_mode = "constant"
    cfg.fixed_penalty = args.lambda_f
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
    suffix = "" if set(contract_names) == set(CONTRACTS) else (
        "_" + "-".join(contract_names))
    cfg.run_name = (f"lambda{args.lambda_f:g}{suffix}_b{block_episodes}"
                    f"{'_epcut' if args.gae_cut_each_episode else ''}"
                    f"{'_bank' if args.state_bank else ''}_s{args.seed}")
    if args.smoke:
        cfg.time_slots = 8
        cfg.update_every = 2
        cfg.eval_episodes = 2
        cfg.min_burn_slots = 20
        cfg.burn_window = 10
        cfg.max_burn_slots = 30
        cfg.checkpoint_every = 0
    cfg = cfg.finalize()

    weights = os.path.join(cfg.out_root, cfg.run_name, "agent.pt")
    if args.stage in ("all", "train"):
        weights = train_joint(cfg, episodes_per_contract, contract_names,
                              block_episodes, args.state_bank, args.resume,
                              gae_cut_each_episode=args.gae_cut_each_episode)
    if args.stage in ("all", "eval"):
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        rows = evaluate_contracts(cfg, weights, contract_names)
        passed = sum(r["valid_full_contract_met"] for r in rows)
        print(f"[joint] valid FullStrict {passed}/{len(rows)}")


if __name__ == "__main__":
    main()
