"""Shared train/evaluate runners used by the scripts."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baselines import edge_only_action, greedy_action, random_action, static_action
from src.config import Params
from src.env import QoSOffloadEnv
from src.ppo import PPOAgent


def make_policy(cfg, env, agent=None, rng=None):
    name = cfg.policy
    if name == "ppo":
        assert agent is not None
        return lambda obs: agent.act(obs, deterministic=False)[:2]
    if name == "random":
        return lambda obs: (random_action(cfg, rng), None)
    if name == "static":
        return lambda obs: (static_action(cfg), None)
    if name == "edge_only":
        return lambda obs: (edge_only_action(cfg), None)
    if name == "greedy":
        return lambda obs: (greedy_action(cfg, env), None)
    raise ValueError(name)


def rollout(cfg, env, policy, deterministic=False, agent=None, reset=True):
    """One rollout chunk. With reset=False the environment continues from its
    current state (tasks, physical queues and virtual queues persist) - the
    P0 chunk semantics: training chunks are segments of ONE continuous
    system run, never fresh episodes.

    For the PPO policy the buffer stores the LATENT action (gates,
    conditional rho, power); the env receives eff_rho = gate*rho with the
    power zeroed for all-local devices (zero-inflated Beta, plan (d))."""
    if reset:
        obs = env.reset()
    else:
        obs = env.current_obs()
    traj = {"obs": [], "acts": [], "logp": [], "rew": [], "val": []}
    logs = []
    for t in range(cfg.time_slots):
        if cfg.policy == "ppo":
            a, logp, v, lat = agent.act(obs, deterministic=deterministic)
        else:
            a, logp = policy(obs)
            v = 0.0
            lat = np.asarray(a, dtype=float)
        traj["obs"].append(obs)
        traj["acts"].append(lat)
        traj["logp"].append(logp if logp is not None else 0.0)
        traj["val"].append(v)
        obs, r, done, info = env.step(a)
        traj["rew"].append(r)
        logs.append(info)
    traj["val"].append(agent.value(obs) if cfg.policy == "ppo" else 0.0)
    return traj, logs


def train(cfg, resume=False):
    rng = np.random.default_rng(cfg.seed)
    torch_seed = cfg.seed
    import torch

    torch.manual_seed(torch_seed)
    np.random.seed(cfg.seed)

    env = QoSOffloadEnv(cfg)
    agent = PPOAgent(cfg)
    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))

    # resume support (review ruling): restore Actor, Critic, both
    # optimizers, env physical queues, Z/Y, RNG states and the training
    # step count so an extended run continues the SAME trajectory
    start_ep = 1
    history = []
    state_path = os.path.join(run_dir, "train_state.pt")
    if resume and os.path.exists(state_path):
        import torch
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        agent.load(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
        env.load_state_dict(ck["env_state"])
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        start_ep = ck["episode"] + 1
        prev_hist = os.path.join(run_dir, "history.json")
        if os.path.exists(prev_hist):
            import json
            with open(prev_hist, encoding="utf-8") as f:
                history = json.load(f)
        print(f"[train {cfg.run_name}] resumed from episode {ck['episode']} "
              f"-> {cfg.episodes}", flush=True)
    # P3: accumulate `update_every` consecutive chunks into ONE trajectory.
    # The env is never reset between chunks, so intermediate chunk ends are
    # NOT episode boundaries: their bootstrap values are dropped and GAE
    # runs across the whole group, bootstrapping only at the final state.
    buf = {"obs": [], "acts": [], "logp": [], "rew": [], "val": []}
    last_stats = {}
    DIAG_KEYS = ("approx_kl", "clipfrac", "entropy", "value_error",
                 "normalized_value_mae", "explained_variance",
                 "critic_clipfrac", "mean_abs_return",
                 "actor_grad_norm", "critic_grad_norm", "boundary_frac",
                 "reward_scale")

    def flush(buffer, bootstrap):
        buffer["val"].append(bootstrap)
        dones = [False] * len(buffer["rew"])
        stats = agent.update(buffer["obs"], buffer["acts"], buffer["logp"],
                             buffer["rew"], dones, buffer["val"])
        for k in buffer:
            buffer[k] = []
        return stats

    for ep in range(start_ep, cfg.episodes + 1):
        # reseed the task stream per chunk for reproducibility, but NEVER
        # reset tasks/queues/virtual queues: one continuous system run
        env.rng = np.random.default_rng(cfg.seed * 100000 + ep)
        policy = make_policy(cfg, env, agent)
        reset = (start_ep == 1 and ep == 1)
        traj, logs = rollout(cfg, env, policy, agent=agent, reset=reset)
        m = env.episode_metrics(logs)
        # P3 ruling: the buffer holds r_train = r_raw / S; GAE and the
        # critic train on r_train, history keeps r_raw. The divisor is the
        # fixed constant cfg.reward_scale (never data-dependent).
        buf["rew"] += [r / cfg.reward_scale for r in traj["rew"]]
        for k in ("obs", "acts", "logp", "val"):
            buf[k] += traj[k][:-1] if k == "val" else traj[k]
        # diagnostics belong to the history row whose samples were trained
        # on; rows without an update carry no (stale) diagnostics
        update_applied = False
        if ep % cfg.update_every == 0:
            last_stats = flush(buf, traj["val"][-1])
            update_applied = True
        entry = {
            "episode": ep,
            "reward": float(np.sum(traj["rew"])),
            "cost": m["cost"],
            "violation": m["violation"],
            "tier_completion": m["tier_completion"].tolist(),
            "tier_norm_delay": m["tier_norm_delay"].tolist(),
            "update_applied": update_applied,
            # chunk-final prices and max compressed price observations:
            # makes the price slope reproducible from history alone
            "Z_final": m["Z_final"].tolist(),
            "Y_final": m["Y_final"].tolist(),
            "z_obs_max": m["z_obs_max"],
            "y_obs_max": m["y_obs_max"],
            # physical stability logs (review ruling)
            "physical_drift": m["physical_drift"],
            "physical_drift_penalty": m["physical_drift_penalty"],
            "local_queue_norm": m["local_queue_norm"],
            "edge_high_queue_norm": m["edge_high_queue_norm"],
            "edge_low_queue_norm": m["edge_low_queue_norm"],
        }
        # offload diagnostics from the LATENT action (gates, conditional rho)
        if cfg.policy == "ppo":
            lat = np.asarray(traj["acts"], dtype=float)
            MK = cfg.devices * cfg.tasks_per_slot
            gates = lat[:, :MK]
            cond_rho = lat[:, MK:2 * MK]
            eff_rho = (gates * cond_rho).reshape(len(lat), cfg.devices, cfg.tasks_per_slot)
            all_local = (gates.reshape(len(lat), cfg.devices, cfg.tasks_per_slot).sum(-1) == 0)
            entry["mean_offload_ratio"] = float(eff_rho.mean())
            # zero-inflated Beta diagnostics (review plan (d))
            entry["offload_gate_rate"] = float(gates.mean())
            entry["conditional_rho_mean"] = float(cond_rho.mean())
            entry["all_local_device_rate"] = float(all_local.mean())
        else:
            # baselines emit the env action directly
            acts_arr = np.asarray(traj["acts"], dtype=float)
            entry["mean_offload_ratio"] = float(
                acts_arr.reshape(len(acts_arr), cfg.devices,
                                 cfg.tasks_per_slot + 1)[:, :, :cfg.tasks_per_slot].mean())
        if update_applied:
            entry["update_index"] = ep
            entry["rollout_samples"] = last_stats["rollout_samples"]
            for k in DIAG_KEYS:
                entry[k] = last_stats[k]
        history.append(entry)

        # periodic checkpoint (review reliability fix): only at update
        # boundaries with an EMPTY buffer, so a resumed run continues from a
        # consistent state; an interruption then loses at most
        # checkpoint_every episodes instead of the whole run
        if (cfg.checkpoint_every and ep % cfg.checkpoint_every == 0
                and ep % cfg.update_every == 0 and not buf["rew"]):
            agent.save(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
            import torch
            torch.save({
                "episode": ep,
                "env_state": env.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
            }, state_path)
            _save_history(history, run_dir)

        if ep % max(1, cfg.episodes // 10) == 0:
            extra = (f" kl={last_stats['approx_kl']:.4f}"
                     f" clip={last_stats['clipfrac']:.3f}"
                     f" bnd={last_stats['boundary_frac']:.2f}"
                     ) if update_applied else ""
            print(f"[train {cfg.run_name}] ep {ep}/{cfg.episodes} "
                  f"reward={entry['reward']:.1f} cost={m['cost']:.3f} "
                  f"viol={m['violation']:.3f}{extra}", flush=True)

    if buf["rew"]:
        # trailing partial group: train it and write the diagnostics back
        # into the LAST history row (the samples just trained on)
        last_stats = flush(buf, traj["val"][-1])
        tail = history[-1]
        tail["update_applied"] = True
        tail["update_index"] = tail["episode"]
        tail["rollout_samples"] = last_stats["rollout_samples"]
        for k in DIAG_KEYS:
            tail[k] = last_stats[k]

    agent.save(os.path.join(run_dir, "agent.pt"), with_optimizers=True)
    import torch
    torch.save({
        "episode": cfg.episodes,
        "env_state": env.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }, state_path)
    _save_history(history, run_dir)
    return history


def _save_history(history, run_dir):
    import json

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f)
    ep = [h["episode"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [h["reward"] for h in history])
    axes[0].set_title("Episode reward")
    axes[1].plot(ep, [h["cost"] for h in history])
    axes[1].set_title("Avg slot cost")
    axes[2].plot(ep, [h["violation"] for h in history])
    axes[2].set_title("Avg slot violation")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def _slope_per_slot(x):
    t = np.arange(len(x), dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(t, np.asarray(x, dtype=float), 1)[0])


def steady_parts(z_slopes, y_slopes, loc_slopes, eh_slopes, el_slopes,
                 z_tol=1e-3, phy_tol=1e-4):
    """Decomposed stationarity (review ruling): the old single flag conflated
    physical-queue stability with QoS-price stability - a contract-violating
    policy keeps Z/Y growing while its PHYSICAL queues are perfectly stable,
    and the two must be reported separately.

    Returns (physical_stationary, price_stationary); both are PER-TIER /
    PER-DEVICE maxima - averages can hide one tier rising while another
    falls."""
    phy = max(float(np.max(np.abs(np.asarray(loc_slopes, dtype=float)))),
              float(np.max(np.abs(np.asarray(eh_slopes, dtype=float)))),
              float(np.max(np.abs(np.asarray(el_slopes, dtype=float)))))
    price = max(float(np.max(np.abs(np.asarray(z_slopes, dtype=float)))),
                float(np.max(np.abs(np.asarray(y_slopes, dtype=float)))))
    return phy <= phy_tol, price <= z_tol


def steady_enough(z_slopes, y_slopes, loc_slopes, eh_slopes, el_slopes,
                  z_tol=1e-3, phy_tol=1e-4):
    """Steady-state burn-in exit condition: PER-TIER / PER-DEVICE slopes all
    inside the tolerance band (physical AND price)."""
    phy, price = steady_parts(z_slopes, y_slopes, loc_slopes, eh_slopes,
                              el_slopes, z_tol, phy_tol)
    return phy and price


def evaluate(cfg, weights_from=None):
    import torch

    torch.manual_seed(cfg.eval_seed)
    rng = np.random.default_rng(cfg.eval_seed)
    env = QoSOffloadEnv(cfg)
    agent = None
    if cfg.policy == "ppo":
        agent = PPOAgent(cfg)
        w_path = weights_from or os.path.join(cfg.out_root, cfg.run_name, "agent.pt")
        agent.load(w_path)
    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    all_metrics = []
    eval_diag = []
    deterministic = (cfg.policy_mode == "deterministic")
    obs = env.reset()
    policy = make_policy(cfg, env, agent, rng)

    def _act_step():
        if cfg.policy == "ppo":
            return agent.act(obs, deterministic=deterministic)[0]
        a, _ = policy(obs)
        return a

    # cold-start segment: the first cold_slots slots after Z/Y = 0, logged
    # separately (review ruling: report cold-start performance)
    cold_logs = []
    cold_slots = min(2000, max(0, cfg.min_burn_slots))
    for _ in range(cold_slots):
        obs, _, _, info = env.step(_act_step())
        cold_logs.append(info)

    # steady-state burn-in: run unlogged until, within the check window,
    # physical queue levels and Z/Y have ~zero slope; min -> max bound;
    # hitting max_burn without stability is reported as non-stationary
    z_traj, y_traj = [], []          # per-tier lists of slot values
    lq_traj = []                     # per-device normalized local queue
    eh_traj, el_traj = [], []
    burn_slots = 0
    stable = False
    check_every = max(1, cfg.burn_window // 10)
    while burn_slots < cfg.max_burn_slots:
        obs, _, _, _ = env.step(_act_step())
        burn_slots += 1
        z_traj.append(env.Z.copy())
        y_traj.append(env.Y.copy())
        lq_traj.append((env.local_q / env.cap_loc).copy())
        eh_traj.append(env.Qe_high / env.cap_high)
        el_traj.append(env.Qe_low / env.cap_low)
        if burn_slots >= cfg.min_burn_slots and burn_slots % check_every == 0:
            w = cfg.burn_window
            z_arr = np.array(z_traj[-w:])          # (w, n_prot)
            y_arr = np.array(y_traj[-w:])
            lq_arr = np.array(lq_traj[-w:])        # (w, M)
            z_sl = [_slope_per_slot(z_arr[:, p]) for p in range(env.n_prot)]
            y_sl = [_slope_per_slot(y_arr[:, p]) for p in range(env.n_prot)]
            lq_sl = [_slope_per_slot(lq_arr[:, i]) for i in range(env.M)]
            eh_sl, el_sl = (_slope_per_slot(eh_traj[-w:]),
                            _slope_per_slot(el_traj[-w:]))
            physical_stationary, price_stationary = steady_parts(
                z_sl, y_sl, lq_sl, [eh_sl], [el_sl])
            stable = physical_stationary and price_stationary
            if stable:
                break
    # final-window slopes are always recorded (decomposed semantics:
    # a contract-violating policy has stable physics but growing prices)
    if burn_slots > 0:
        w = min(cfg.burn_window, burn_slots)
        z_arr = np.array(z_traj[-w:]); y_arr = np.array(y_traj[-w:])
        lq_arr = np.array(lq_traj[-w:])
        z_sl = [_slope_per_slot(z_arr[:, p]) for p in range(env.n_prot)]
        y_sl = [_slope_per_slot(y_arr[:, p]) for p in range(env.n_prot)]
        lq_sl = [_slope_per_slot(lq_arr[:, i]) for i in range(env.M)]
        eh_sl, el_sl = (_slope_per_slot(eh_traj[-w:]),
                        _slope_per_slot(el_traj[-w:]))
        physical_stationary, price_stationary = steady_parts(
            z_sl, y_sl, lq_sl, [eh_sl], [el_sl])
    else:
        z_sl = y_sl = lq_sl = []
        eh_sl = el_sl = 0.0
        physical_stationary = price_stationary = False

    for ep in range(cfg.eval_episodes):
        env.rng = np.random.default_rng(cfg.eval_seed * 100000 + ep)
        traj, logs = rollout(cfg, env, policy, deterministic=deterministic,
                             agent=agent, reset=False)
        all_metrics.append(env.episode_metrics(logs))
        if cfg.policy == "ppo":
            lat = np.asarray(traj["acts"], dtype=float)
            MK = cfg.devices * cfg.tasks_per_slot
            gates, cond_rho = lat[:, :MK], lat[:, MK:2 * MK]
            eval_diag.append({
                "offload_gate_rate": float(gates.mean()),
                "conditional_rho_mean": float(cond_rho.mean()),
                "all_local_device_rate": float(
                    (gates.reshape(len(lat), cfg.devices, cfg.tasks_per_slot).sum(-1) == 0).mean()),
            })

    keys = ["cost", "energy", "tx_energy", "energy_total", "tx_bits",
            "edge_q", "violation"]
    summary = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    for key in ["tier_completion", "tier_norm_delay", "tier_max_norm_delay",
                "tier_starvation_rate", "tier_utility", "tier_p95_norm_delay",
                "z_obs_max", "y_obs_max"]:
        arr = np.stack([m[key] for m in all_metrics])
        summary[key] = arr.mean(axis=0).tolist()
        # chunks of one continuous run are correlated: this is a spread over
        # chunks, NOT an independent-replication std (do not report as se)
        summary[key + "_chunk_std"] = arr.std(axis=0).tolist()
    for key in ["Z_final", "Y_final"]:
        summary[key] = np.stack([m[key] for m in all_metrics]).mean(axis=0).tolist()
    # raw per-task normalized-delay samples (E5 tail / CCDF plots);
    # edge-branch-only samples scope-match the T5/T5+ pool-wait statements
    samples = {}
    edge_samples = {}
    for p in range(cfg.tiers):
        chunks = [np.asarray(m["tier_delay_samples"][p], dtype=float)
                  for m in all_metrics]
        chunks = [x for x in chunks if x.size]
        if chunks:
            samples[f"tier{p + 1}"] = np.concatenate(chunks)
        echunks = [np.asarray(m["tier_edge_delay_samples"][p], dtype=float)
                   for m in all_metrics]
        echunks = [x for x in echunks if x.size]
        if echunks:
            edge_samples[f"tier{p + 1}"] = np.concatenate(echunks)
    if samples:
        np.savez_compressed(os.path.join(run_dir, "delay_samples.npz"), **samples)
    if edge_samples:
        np.savez_compressed(os.path.join(run_dir, "edge_wait_samples.npz"), **edge_samples)
    summary["reserve_alpha_effective"] = float(env.reserve_alpha)
    summary["aging_mode"] = cfg.aging_mode
    summary["beta_0"] = cfg.beta_0
    summary["kappa_a"] = cfg.kappa_a
    # cold-start segment statistics (from Z/Y = 0) and burn-in metadata
    if cold_logs:
        cold = QoSOffloadEnv(cfg).episode_metrics(cold_logs)
        summary["cold_start"] = {
            "slots": len(cold_logs),
            "tier_completion": cold["tier_completion"].tolist(),
            "tier_norm_delay": cold["tier_norm_delay"].tolist(),
            "violation": cold["violation"],
        }
    cold_slots_done = len(cold_logs)
    summary["cold_start_slots"] = cold_slots_done
    summary["post_cold_burn_slots"] = burn_slots
    summary["total_burn_slots"] = cold_slots_done + burn_slots
    summary["burn_stable"] = bool(stable)
    # decomposed stationarity (review ruling): physical vs QoS-price
    summary["physical_stationary"] = bool(physical_stationary)
    summary["price_stationary"] = bool(price_stationary)
    summary["contract_stationary"] = bool(physical_stationary and price_stationary)
    summary["price_z_slopes_per_tier"] = [float(x) for x in z_sl]
    summary["price_y_slopes_per_tier"] = [float(x) for x in y_sl]
    summary["local_queue_slopes_per_device"] = [float(x) for x in lq_sl]
    summary["edge_high_slope"] = float(eh_sl)
    summary["edge_low_slope"] = float(el_sl)
    summary["non_stationary"] = bool(not stable and burn_slots >= cfg.max_burn_slots)
    # AL multiplier boundary fraction (probe gate 5): share of logged slots
    # where any multiplier sits at 0 or lambda_max
    lam_b = [x.get("lam_at_boundary") for x in logs
             if x.get("lam_at_boundary") is not None]
    if lam_b:
        summary["lam_boundary_frac"] = float(np.mean(lam_b))
    summary["policy_mode"] = cfg.policy_mode
    summary["run_name"] = cfg.run_name
    summary["reward_mode"] = cfg.reward_mode
    summary["policy"] = cfg.policy
    summary["eval_episodes"] = cfg.eval_episodes
    if eval_diag:
        for k in eval_diag[0]:
            summary[k] = float(np.mean([d[k] for d in eval_diag]))
    import json

    with open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval {cfg.run_name}] cost={summary['cost']:.3f} "
          f"violation={summary['violation']:.4f} "
          f"completion={np.round(summary['tier_completion'], 3).tolist()} "
          f"starvation={np.round(summary['tier_starvation_rate'], 3).tolist()} "
          f"utility={np.round(summary['tier_utility'], 3).tolist()}")
    return summary
