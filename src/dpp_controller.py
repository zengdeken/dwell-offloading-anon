"""Non-learning per-slot DPP controller (route 2, review ruling).

Minimizes the exact one-slot drift-plus-penalty objective each slot:

    G(a) = V*cost(a) + sum_p Z_p*g_z,p(a) + sum_p Y_p*g_y,p(a)
           + lambda_phy * dLphy(a)

which is EXACTLY the env's r_raw negated (reward_mode="vq"): the controller
and the theory (T3' with a measured xi) share one objective.

Candidate evaluation uses FULL ENV STATE SNAPSHOTS (state_dict /
load_state_dict capture queues, prices, current tasks, channel AND the RNG
bit-generator state), so every candidate is simulated from the identical
state and sees the identical downstream random stream - candidate order
cannot change the task flow (review requirement).

Search: coordinate descent over per-task offload ratio in
{0, 0.25, 0.5, 0.75, 1} (0 = exact all-local, subsuming the ZIB gate) and
per-device power in {0.25, 0.5, 0.75, 1}, `rounds` full sweeps, warm-started
from the previous slot's action. The controller reads ONLY the contract
(cfg.eta_min / t_max are baked into the env prices' gz/gy via the step) and
the current state - no training, no per-contract tuning.
"""

import time

import numpy as np


class DPPCoordinateController:
    def __init__(self, env, rho_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                 power_grid=(0.25, 0.5, 0.75, 1.0), rounds=2):
        self.env = env
        self.rho_grid = list(rho_grid)
        self.power_grid = list(power_grid)
        self.rounds = int(rounds)
        cfg = env.cfg
        self.M, self.K = cfg.devices, cfg.tasks_per_slot
        # warm start near the capacity-driven stable point (grid-snapped)
        r0 = min(self.rho_grid, key=lambda x: abs(x - cfg.rho_init))
        p0 = 0.5
        initial = np.full((self.M, self.K + 1), r0, dtype=float)
        initial[:, self.K] = p0
        self.last = initial.reshape(-1)
        self.solve_times = []
        self.g_trace = []          # per-slot round-wise G improvements

    def _eval_G(self, action, state, z_pre, y_pre, lam, V):
        """Simulate one candidate from the SAME state; return G = -r_raw."""
        self.env.load_state_dict(state)
        _, r, _, info = self.env.step(action)
        # r = -(V*cost + Z_pre.gz + Y_pre.gy + lam*drift) in vq mode
        return -r, info

    def act(self):
        """One slot: coordinate search; returns the chosen env action."""
        t0 = time.perf_counter()
        env = self.env
        V, lam = env.cfg.V, env.cfg.lambda_phy
        state = env.state_dict()
        z_pre, y_pre = env.Z.copy(), env.Y.copy()

        best = self.last.copy()
        best_G, _ = self._eval_G(best, state, z_pre, y_pre, lam, V)
        g_by_round = [best_G]
        for _ in range(self.rounds):
            # task offload coordinates
            for i in range(self.M):
                for k in range(self.K):
                    j = i * (self.K + 1) + k
                    cur = best[j]
                    for rho in self.rho_grid:
                        if rho == cur:
                            continue
                        cand = best.copy()
                        cand[j] = rho
                        G, _ = self._eval_G(cand, state, z_pre, y_pre, lam, V)
                        if G < best_G - 1e-12:
                            best_G, best = G, cand
            # device power coordinates
            for i in range(self.M):
                j = i * (self.K + 1) + self.K
                cur = best[j]
                for pw in self.power_grid:
                    if pw == cur:
                        continue
                    cand = best.copy()
                    cand[j] = pw
                    G, _ = self._eval_G(cand, state, z_pre, y_pre, lam, V)
                    if G < best_G - 1e-12:
                        best_G, best = G, cand
            g_by_round.append(best_G)

        # restore the true state before returning (caller steps for real)
        env.load_state_dict(state)
        self.last = best
        self.solve_times.append(time.perf_counter() - t0)
        # empirical xi proxy: improvement still available in the last round
        self.g_trace.append(g_by_round)
        return best

    def xi_proxy(self):
        """Last-round improvement (0 when converged): the measured per-slot
        suboptimality gap of the coordinate search for T3'."""
        if not self.g_trace:
            return 0.0
        gaps = [tr[-2] - tr[-1] for tr in self.g_trace if len(tr) >= 2]
        return float(np.mean(gaps)) if gaps else 0.0
