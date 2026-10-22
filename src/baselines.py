"""Non-learning baseline policies for the differentiated-QoS offloading env."""

import numpy as np


def random_action(cfg, rng):
    return rng.uniform(0, 1, cfg.devices * (cfg.tasks_per_slot + 1))


def static_action(cfg):
    """Fixed half offloading, 0.7 power (still enjoys priority scheduling)."""
    a = np.full(cfg.devices * (cfg.tasks_per_slot + 1), 0.5, dtype=float)
    a.reshape(cfg.devices, cfg.tasks_per_slot + 1)[:, -1] = 0.7
    return a


def edge_only_action(cfg):
    a = np.full(cfg.devices * (cfg.tasks_per_slot + 1), 1.0, dtype=float)
    return a


def greedy_action(cfg, env):
    """Myopic tier-ordered greedy: for each task (tier desc), pick the
    offload ratio from a candidate set minimizing immediate normalized cost
    plus a delay-tightness penalty, respecting the transmission budget.

    The score mirrors the env physics (review item 3):
      - tx energy P*T_tx is part of the task cost estimate, so high power
        is no longer free (no systematic max-power bias);
      - edge waiting is pool-aware (Qe_high vs Qe_low at their own rates)
        and includes the end-of-slot queue-entry delay delta (causality).
    The power ratio is searched over a small grid with the same score.
    """
    K, M, L = cfg.tasks_per_slot, cfg.devices, cfg.tiers
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    powers = [0.5, 0.7, 1.0]
    cap_e = cfg.edge_freq * cfg.delta
    # guaranteed pool rates - identical to the env's departure estimate
    mu_high = env.reserve_alpha * cfg.edge_freq
    mu_low = (1.0 - env.reserve_alpha) * cfg.edge_freq
    best_act, best_total = None, None
    for power in powers:
        act = np.zeros(M * (K + 1))
        total = 0.0
        for i in range(M):
            act[i * (K + 1) + K] = power
            rate = env.tx_rate_mbps(i, power)   # same wireless model as env
            budget = max(0.0, rate * cfg.delta)
            p_w = env.trans_powers_w[i] * power
            order = sorted(range(K), key=lambda k: -int(env.task_tier[i, k]))
            for k in order:
                d, cc, tau = env.task_d[i, k], env.task_c[i, k], env.task_tau[i, k]
                tier = int(env.task_tier[i, k])
                mu_pool = mu_high if tier == L - 1 else mu_low
                best, best_ratio = None, 0.0
                for r in candidates:
                    got = min(r * d, budget)
                    l_load = (d - got) * 1e6 * cc
                    w_e = got * 1e6 * cc
                    # crude completion estimate: local queue + own load over f_i
                    t_loc = (env.local_q[i] + l_load) / env.comp_freqs[i]
                    # edge branch exists only if work is actually sent:
                    # entry at end of slot, behind own pool's backlog,
                    # own service at the POOL rate (not the raw edge rate)
                    if got > 1e-12:
                        pool_wait = (env.Qe_high / mu_high if tier == L - 1
                                     else env.Qe_low / mu_low)
                        t_edge = cfg.delta + pool_wait + w_e / mu_pool
                    else:
                        t_edge = 0.0
                    e_tx = p_w * (got / rate) if rate > 1e-12 else 0.0
                    e_ref = cfg.energy_coef * d * 1e6 * cc
                    expn_ref = cfg.service_price * d * 1e6 * cc
                    e_task = cfg.energy_coef * l_load + e_tx
                    c_est = (cfg.energy_weight * min(e_task / max(e_ref, 1e-12), 1.0)
                             + cfg.expense_weight
                             * min(cfg.service_price * w_e / max(expn_ref, 1e-12), 1.0))
                    w = 1.0 + 2.0 * tier                  # tier weight
                    score = w * c_est + 5.0 * max(t_loc, t_edge) / tau
                    if best is None or score < best:
                        best, best_ratio = score, r
                used = min(best_ratio * d, budget)
                budget -= used
                act[i * (K + 1) + k] = best_ratio
                total += best
        if best_total is None or total < best_total:
            best_act, best_total = act, total
    return best_act
