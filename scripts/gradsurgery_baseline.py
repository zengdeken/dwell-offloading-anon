"""Gradient-surgery multi-task baseline (Yu et al., NeurIPS 2020).

Same budget and evaluation protocol as the joint fixed-penalty baseline,
but the training mix is the multi-task regime the letter argued cannot
work: the contract is re-sampled i.i.d. EVERY episode (fresh env reset per
episode, done at every episode end), and the actor gradient is formed by
PCGrad - per-contract gradients are computed on each minibatch and
conflicting components are projected out before the sum. The critic is
trained on the mixed batch.

This measures whether gradient surgery can recover the multi-contract
regime that per-episode switching starves of steady-state data.
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
from src.ppo import PPOAgent, compute_gae
from src.runner import _save_history, evaluate, make_policy, rollout


def atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class SurgeryPPOAgent(PPOAgent):
    """PPOAgent whose actor update forms the minibatch gradient by PCGrad
    over per-contract sub-batches (conflict projection + sum)."""

    def update(self, obs, acts, logp_old, rewards, dones, values,
               contract_ids=None):
        c = self.cfg
        T = len(rewards)
        adv, returns = compute_gae(rewards, values, dones, c.gamma, c.lam)
        adv_t = torch.as_tensor(
            (adv - adv.mean()) / (adv.std() + 1e-5), dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.asarray(acts), dtype=torch.float32)
        logp_old_t = torch.as_tensor(np.asarray(logp_old), dtype=torch.float32)
        values_t = torch.as_tensor(np.asarray(values[:-1]), dtype=torch.float32)
        cid_t = (torch.as_tensor(np.asarray(contract_ids, dtype=np.int64))
                 if contract_ids is not None else torch.zeros(T, dtype=torch.int64))
        n_tasks = len(CONTRACTS)

        n = T
        mb = min(c.batch_size, n)
        params = list(self.actor.parameters())

        kl_acc, clip_acc, ent_acc, p_loss_acc, gn_policy = [], [], [], [], []
        p_weight, p_updates = 0, 0
        kl_early_stop = False
        for _ in range(c.p_epochs):
            epoch_kl, epoch_w = 0.0, 0.0
            perm = torch.randperm(n)
            for start in range(0, n, mb):
                ids = perm[start:start + mb]
                logp, ent = self._evaluate(obs_t[ids], act_t[ids])
                logratio = logp - logp_old_t[ids]
                ratio = torch.exp(logratio)
                s1 = ratio * adv_t[ids]
                s2 = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv_t[ids]

                # ---- per-contract gradients on this minibatch ----
                flat_grads = []
                ids_c = cid_t[ids]
                for gid in range(n_tasks):
                    mask = ids_c == gid
                    if not mask.any():
                        continue
                    p_loss_g = -(torch.min(s1[mask], s2[mask])
                                 + c.entropy * ent[mask]).mean()
                    grads = torch.autograd.grad(p_loss_g, params,
                                                retain_graph=True)
                    flat_grads.append(torch.cat(
                        [g.reshape(-1) for g in grads]))
                if not flat_grads:
                    continue
                # ---- PCGrad: project conflicts in a random order ----
                order = torch.randperm(len(flat_grads)).tolist()
                projected = [g.clone() for g in flat_grads]
                for ii in order:
                    for jj in order:
                        if ii == jj:
                            continue
                        dot = torch.dot(projected[ii], flat_grads[jj])
                        if dot < 0:
                            projected[ii] = (projected[ii]
                                             - dot / (torch.dot(
                                                 flat_grads[jj],
                                                 flat_grads[jj]) + 1e-9)
                                             * flat_grads[jj])
                total = sum(projected)
                with torch.no_grad():
                    idx = 0
                    for p in params:
                        numel = p.numel()
                        p.grad = total[idx:idx + numel].view_as(p).clone()
                        idx += numel
                gnorm = torch.nn.utils.clip_grad_norm_(
                    params, c.actor_grad_clip)
                self.actor_opt.step()
                p_updates += 1

                with torch.no_grad():
                    logp_new, ent_new = self._evaluate(obs_t[ids], act_t[ids])
                    logratio_new = logp_new - logp_old_t[ids]
                    ratio_new = torch.exp(logratio_new)
                    kl = (ratio_new - 1.0 - logratio_new).mean()
                    clipfrac = ((ratio_new - 1.0).abs() > c.clip).float().mean()
                b = len(ids)
                p_weight += b
                epoch_kl += float(kl) * b
                epoch_w += b
                kl_acc.append(float(kl) * b)
                clip_acc.append(float(clipfrac) * b)
                ent_acc.append(float(ent.mean()) * b)
                gn_policy.append(float(gnorm) * b)
            if c.target_kl > 0.0 and epoch_w > 0 and epoch_kl / epoch_w > c.target_kl:
                kl_early_stop = True
                break

        v_loss_acc, ve_acc, gn_critic = [], [], []
        v_weight, v_updates = 0, 0
        for _ in range(c.v_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, mb):
                ids = perm[start:start + mb]
                v = self.critic(obs_t[ids]).squeeze(-1)
                v_loss = torch.nn.functional.mse_loss(v, ret_t[ids])
                self.critic_opt.zero_grad()
                v_loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), c.critic_grad_clip)
                self.critic_opt.step()
                v_updates += 1
                with torch.no_grad():
                    ve = (v - ret_t[ids]).abs().mean()
                b = len(ids)
                v_weight += b
                v_loss_acc.append(float(v_loss) * b)
                ve_acc.append(float(ve) * b)
                gn_critic.append(float(gnorm) * b)

        ret_np = ret_t.numpy()
        value_error = float(np.sum(ve_acc) / v_weight) if v_weight else 0.0
        stats = {
            "p_loss": float(np.sum(p_loss_acc) / p_weight) if p_weight else 0.0,
            "v_loss": float(np.sum(v_loss_acc) / v_weight) if v_weight else 0.0,
            "entropy": float(np.sum(ent_acc) / p_weight) if p_weight else 0.0,
            "approx_kl": float(np.sum(kl_acc) / p_weight) if p_weight else 0.0,
            "clipfrac": float(np.sum(clip_acc) / p_weight) if p_weight else 0.0,
            "value_error": value_error,
            "explained_variance": 0.0,
            "critic_clipfrac": 0.0,
            "mean_abs_return": float(np.mean(np.abs(ret_np))),
            "actor_grad_norm": float(np.sum(gn_policy) / p_weight) if p_weight else 0.0,
            "critic_grad_norm": float(np.sum(gn_critic) / v_weight) if v_weight else 0.0,
            "boundary_frac": 0.0,
            "p_updates": p_updates, "v_updates": v_updates,
            "p_epochs_run": c.p_epochs, "kl_early_stop": kl_early_stop,
            "target_kl": float(c.target_kl), "p_weight": p_weight,
            "v_weight": v_weight, "reward_scale": float(c.reward_scale),
            "rollout_samples": T,
        }
        return stats


def apply_contract(env, name):
    spec = CONTRACTS[name]
    env.set_contract(spec["eta_min"], spec["t_max"])


def train_surgery(cfg, episodes_per_contract, contract_names, resume=False):
    total_episodes = episodes_per_contract * len(contract_names)
    cfg.episodes = total_episodes
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = QoSOffloadEnv(cfg)
    agent = SurgeryPPOAgent(cfg)
    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    atomic_json(os.path.join(run_dir, "surgery_protocol.json"), {
        "method": "PCGrad gradient surgery, i.i.d. per-episode contracts",
        "contracts": {name: CONTRACTS[name] for name in contract_names},
        "episodes_per_contract": episodes_per_contract,
        "total_episodes": total_episodes,
        "reset_each_episode": True,
        "pcgrad": True,
    })

    buf = {"obs": [], "acts": [], "logp": [], "rew": [],
           "val": [], "dones": [], "cids": []}
    history = []

    def flush(bootstrap):
        values = buf["val"] + [bootstrap]
        stats = agent.update(buf["obs"], buf["acts"], buf["logp"],
                             buf["rew"], buf["dones"], values,
                             contract_ids=buf["cids"])
        for key in buf:
            buf[key] = []
        return stats

    for ep in range(1, total_episodes + 1):
        rng = np.random.default_rng(cfg.seed * 100000 + ep)
        cname = str(rng.choice(contract_names))
        apply_contract(env, cname)
        env.reset()
        policy = make_policy(cfg, env, agent)
        traj, logs = rollout(cfg, env, policy, agent=agent, reset=False)
        metrics = env.episode_metrics(logs)

        cid = contract_names.index(cname)
        buf["obs"].extend(traj["obs"])
        buf["acts"].extend(traj["acts"])
        buf["logp"].extend(traj["logp"])
        buf["rew"].extend(r / cfg.reward_scale for r in traj["rew"])
        buf["val"].extend(traj["val"][:-1])
        ep_dones = [False] * len(traj["rew"])
        if ep < total_episodes:
            ep_dones[-1] = True
        buf["dones"].extend(ep_dones)
        buf["cids"].extend([cid] * len(traj["rew"]))
        last_bootstrap = traj["val"][-1]

        update_applied = False
        if ep % cfg.update_every == 0:
            last_stats = flush(last_bootstrap)
            update_applied = True

        spec = CONTRACTS[cname]
        compliance = contract_metrics(
            metrics["tier_completion"], metrics["tier_norm_delay"],
            spec["eta_min"], spec["t_max"], None)
        history.append({
            "episode": ep, "contract": cname,
            "reward": float(np.sum(traj["rew"])),
            "cost": metrics["cost"], "violation": metrics["violation"],
            "tier_completion": metrics["tier_completion"].tolist(),
            "tier_norm_delay": metrics["tier_norm_delay"].tolist(),
            "full_contract_met": compliance["full_contract_met"],
            "eta_margin": compliance["eta_margin"],
            "delay_margin": compliance["delay_margin"],
            "update_applied": update_applied,
        })

        if ep % 50 == 0:
            _save_history(history, run_dir)
        if ep % max(1, total_episodes // 10) == 0:
            print(f"[surgery {cfg.run_name}] {ep}/{total_episodes} {cname} "
                  f"cost={metrics['cost']:.3f} "
                  f"viol={metrics['violation']:.3f}", flush=True)

    if buf["rew"]:
        flush(last_bootstrap)
        history[-1]["update_applied"] = True

    agent.save(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
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
            "method": "gradsurgery",
            **compliance,
        })
    out = os.path.join(cfg.out_root, parent_run, "joint_eval.json")
    atomic_json(out, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_f", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--episodes_per_contract", type=int, default=1000)
    ap.add_argument("--time_slots", type=int, default=100)
    ap.add_argument("--update_every", type=int, default=2)
    ap.add_argument("--out_root", default="results/gradsurgery")
    ap.add_argument("--stage", choices=["all", "train", "eval"], default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    contract_names = sorted(CONTRACTS)
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
    if args.smoke:
        episodes_per_contract = 1
    cfg.run_name = f"pcgrad_lambda{args.lambda_f:g}_s{args.seed}"
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
        weights = train_surgery(cfg, episodes_per_contract, contract_names)
    if args.stage in ("all", "eval"):
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        rows = evaluate_contracts(cfg, weights, contract_names)
        viol = [r["violation"] for r in rows]
        print(f"[surgery summary] mean violation {np.mean(viol):.4f} "
              f"over {len(rows)} contracts")


if __name__ == "__main__":
    main()
