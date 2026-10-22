"""Penalty-free lexicographic contract-debt action filter."""

import time

import numpy as np

from src.baselines import greedy_action


class ContractDebtFilter:
    """Select a low-cost action subject to non-growing predicted debt.

    The first action is selected from a finite structured candidate set. For
    the remaining horizon, a fixed deterministic policy supplies actions.
    Candidate rollouts share common random-number scenario states.
    """

    def __init__(self, env, agent, horizon=3, scenarios=4,
                 candidate_count=12, seed=9100):
        self.env = env
        self.agent = agent
        self.horizon = int(horizon)
        self.scenarios = int(scenarios)
        self.candidate_count = int(candidate_count)
        self.seed = int(seed)
        self.step_index = 0
        self.solve_times = []
        self.telemetry = []

    def _base_action(self):
        return self.agent.act(
            self.env.current_obs(), deterministic=True)[0]

    def _candidate_actions(self, base):
        cfg = self.env.cfg
        matrix = np.asarray(base, dtype=float).reshape(
            cfg.devices, cfg.tasks_per_slot + 1)
        candidates = [
            matrix.copy(),
            np.asarray(greedy_action(cfg, self.env), dtype=float).reshape(
                cfg.devices, cfg.tasks_per_slot + 1),
            np.ones_like(matrix),
            np.zeros_like(matrix),
        ]
        candidates[-1][:, -1] = 0.0

        for scale in (0.75, 1.25):
            action = matrix.copy()
            action[:, :cfg.tasks_per_slot] = np.clip(
                action[:, :cfg.tasks_per_slot] * scale, 0.0, 1.0)
            candidates.append(action)
        for scale in (0.75, 1.25):
            action = matrix.copy()
            action[:, -1] = np.clip(action[:, -1] * scale, 0.0, 1.0)
            candidates.append(action)
        for tier in range(cfg.protected_low):
            mask = self.env.task_tier == tier
            for delta in (-0.25, 0.25):
                action = matrix.copy()
                task_action = action[:, :cfg.tasks_per_slot]
                task_action[mask] = np.clip(task_action[mask] + delta,
                                            0.0, 1.0)
                candidates.append(action)

        unique = []
        seen = set()
        for action in candidates:
            flat = np.clip(action.reshape(-1), 0.0, 1.0)
            key = tuple(np.round(flat, 8))
            if key not in seen:
                seen.add(key)
                unique.append(flat)
        return unique[:self.candidate_count]

    def _scenario_states(self, base_state):
        # Scenario 0 is the exact realized RNG stream. Remaining scenarios
        # perturb only future randomness while preserving current tasks/state.
        states = [base_state]
        for scenario in range(1, self.scenarios):
            self.env.load_state_dict(base_state)
            self.env.rng = np.random.default_rng(
                self.seed + self.step_index * 100003 + scenario)
            states.append(self.env.state_dict())
        self.env.load_state_dict(base_state)
        return states

    def _debt_limits(self, z_pre, y_pre):
        cfg = self.env.cfg
        dead_z = np.full(self.env.n_prot, cfg.eps_z, dtype=float)
        dead_y = cfg.eps_y * np.maximum(self.env.t_max, 1.0)
        target_z = np.maximum(dead_z, z_pre - 0.5 * dead_z)
        target_y = np.maximum(dead_y, y_pre - 0.5 * dead_y)
        scale_z = np.maximum(dead_z, cfg.eps_z)
        scale_y = np.maximum(dead_y, cfg.eps_y)
        return target_z, target_y, scale_z, scale_y

    def _evaluate_candidate(self, action, scenario_states, limits):
        target_z, target_y, scale_z, scale_y = limits
        costs, risks = [], []
        terminal_z, terminal_y = [], []
        for state in scenario_states:
            self.env.load_state_dict(state)
            horizon_cost = 0.0
            for step in range(self.horizon):
                chosen = action if step == 0 else self._base_action()
                _, _, _, info = self.env.step(chosen)
                horizon_cost += float(info["cost"])
            z = self.env.Z.copy()
            y = self.env.Y.copy()
            risk = max(float(np.max((z - target_z) / scale_z)),
                       float(np.max((y - target_y) / scale_y)))
            costs.append(horizon_cost / self.horizon)
            risks.append(risk)
            terminal_z.append(z)
            terminal_y.append(y)
        return {
            "action": np.asarray(action, dtype=float),
            "risk": float(np.max(risks)),
            "cost": float(np.mean(costs)),
            "terminal_z": np.mean(terminal_z, axis=0),
            "terminal_y": np.mean(terminal_y, axis=0),
        }

    def act(self):
        started = time.perf_counter()
        base_state = self.env.state_dict()
        z_pre, y_pre = self.env.Z.copy(), self.env.Y.copy()
        base = self._base_action()
        candidates = self._candidate_actions(base)
        scenarios = self._scenario_states(base_state)
        limits = self._debt_limits(z_pre, y_pre)

        rows = [self._evaluate_candidate(a, scenarios, limits)
                for a in candidates]
        safe = [row for row in rows if row["risk"] <= 0.0]
        fallback = not safe
        chosen = (min(safe, key=lambda row: row["cost"])
                  if safe else min(rows, key=lambda row: (row["risk"],
                                                          row["cost"])))
        self.env.load_state_dict(base_state)
        elapsed = time.perf_counter() - started
        self.solve_times.append(elapsed)
        self.telemetry.append({
            "fallback": fallback,
            "safe_candidates": len(safe),
            "candidate_count": len(rows),
            "risk": chosen["risk"],
            "predicted_cost": chosen["cost"],
            "z_pre": z_pre.tolist(),
            "y_pre": y_pre.tolist(),
            "z_terminal": chosen["terminal_z"].tolist(),
            "y_terminal": chosen["terminal_y"].tolist(),
            "solve_time_ms": elapsed * 1000.0,
        })
        self.step_index += 1
        return chosen["action"]

    def summary(self):
        if not self.telemetry:
            return {}
        times = np.asarray(self.solve_times, dtype=float) * 1000.0
        return {
            "steps": len(self.telemetry),
            "fallback_rate": float(np.mean(
                [row["fallback"] for row in self.telemetry])),
            "safe_candidates_mean": float(np.mean(
                [row["safe_candidates"] for row in self.telemetry])),
            "risk_mean": float(np.mean(
                [row["risk"] for row in self.telemetry])),
            "solve_time_mean_ms": float(times.mean()),
            "solve_time_p95_ms": float(np.percentile(times, 95)),
        }
