"""Differentiated-QoS MEC partial-offloading environment (P0 physics).

Three-stage model per slot: transmission budget, local Lindley queue,
edge reservation pools. P0 correctness contract (review v2.1):

  1. Work conservation: D_edge = min(rho*D, actual transmission), and
     D_local = D - D_edge. Every transmitted bit is billed (service price,
     tx time, edge work); every untransmitted bit stays local. No phantom
     work: a per-slot ledger (arrivals, pre/post backlog, served) is kept
     for local, edge-high and edge-low streams and satisfies
         q_pre + arrivals == served + q_post        (machine precision)
  2. Edge backlog occupies future capacity: per-pool Lindley form
         Qe(t+1) = max(Qe(t) + Ae(t) - budget, 0)
     with SEPARATE high/low pools (reservation: high pool capped at
     alpha_e*C_e, unused reservation returned to the low pool within the
     slot; the low pool never lends upward). Old backlog delays new tasks
     through Qe_pre/mu_pool in the departure formula.
  3. Wireless model in SI units: SNR = P[h = path gain] / (N0*B),
     R = B*log2(1+SNR); the observation is the normalized log-rate
     log2(1+SNR)/log2(1+SNR_ref), monotone and channel-distinguishable
     (no saturation clipping).
  4. Transmission energy E_tx = P * T_tx is part of the task cost, so the
     power action has a price.

Task semantics remain per-slot cohorts: a task's branch delay is its
drain position in the (persistent) queue; the residual work carries to
the next slot through the Lindley update.

Mechanism layer (v1.1, unchanged): top-tier structural priority,
price-driven class-level aging inside the low pool, absent-tier masking
of the virtual queues, utility semantics.
"""

import numpy as np


def utility_value(tbar, a_p):
    """Piecewise satisfaction utility (opening-report Eq.(4)).

    U = 1 for Tbar <= a_p, linear decay to 0 at Tbar = 1 (deadline), 0 beyond.
    """
    tbar = np.asarray(tbar, dtype=float)
    if tbar.ndim == 0:
        if tbar <= a_p:
            return 1.0
        if tbar >= 1.0:
            return 0.0
        return float(1.0 - (tbar - a_p) / max(1e-12, 1.0 - a_p))
    u = np.ones_like(tbar)
    decay = (tbar > a_p) & (tbar < 1.0)
    u[decay] = 1.0 - (tbar[decay] - a_p) / max(1e-12, 1.0 - a_p)
    u[tbar >= 1.0] = 0.0
    return u


class QoSOffloadEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg
        self.rng = np.random.default_rng(c.seed)

        self.M, self.K, self.L = c.devices, c.tasks_per_slot, c.tiers
        self.n_prot = c.protected_low

        # ---- wireless model, SI units ----
        self.bandwidth_hz = c.bandwidth                    # Hz per device
        self.trans_powers = np.asarray(c.trans_powers[:c.devices], dtype=float)
        self.path_loss = np.asarray(c.path_loss[:c.devices], dtype=float)
        self.comp_freqs = np.asarray(c.comp_freqs[:c.devices], dtype=float)
        if len(self.trans_powers) < c.devices:
            self.trans_powers = np.resize(self.trans_powers, c.devices)
            self.path_loss = np.resize(self.path_loss, c.devices)
            self.comp_freqs = np.resize(self.comp_freqs, c.devices)
        self.trans_powers_w = self.trans_powers / 1000.0   # mW -> W
        # N0 in W/Hz (cfg.finalize converts dBm/Hz), noise power = N0 * B
        self.noise_watt = c.noise_power * self.bandwidth_hz
        # reference SNR for the log-scale observation (strongest device,
        # full power); obs rate ratio is monotone in (P, h) and unsaturated
        self.snr_ref = float(np.max(self.trans_powers_w * self.path_loss)
                             / self.noise_watt)

        self.eta_min = np.asarray(c.eta_min[:self.n_prot], dtype=float)
        self.t_max = np.asarray(c.t_max[:self.n_prot], dtype=float)
        if len(self.eta_min) < self.n_prot:
            self.eta_min = np.resize(self.eta_min, self.n_prot)
            self.t_max = np.resize(self.t_max, self.n_prot)

        # task samplers
        self.d_lo = c.data_size_kb[0] * 1024 * 8 * 1e-6     # Mb
        self.d_hi = c.data_size_kb[1] * 1024 * 8 * 1e-6
        self.c_lo = c.comp_dens[0] * 1e-9                   # Gcycles/bit
        self.c_hi = c.comp_dens[1] * 1e-9
        self.tier_probs = np.asarray(c.tier_probs, dtype=float)
        self.tier_probs = self.tier_probs / self.tier_probs.sum()

        # state
        self.local_q = np.zeros(c.devices)                  # Gcycles
        self.Qe_high = 0.0                                  # Gcycles (top tier)
        self.Qe_low = 0.0                                   # Gcycles (low pool)
        self.channel = np.ones(c.devices)
        # tasks per device: arrays [K] each
        self.task_d = np.zeros((c.devices, self.K))
        self.task_c = np.zeros((c.devices, self.K))
        self.task_tau = np.zeros((c.devices, self.K))
        self.task_tier = np.zeros((c.devices, self.K), dtype=int)

        # virtual queues + last-slot tier stats
        self.Z = np.zeros(self.n_prot)
        self.Y = np.zeros(self.n_prot)
        # AL multipliers (DESIGN_FREEZE_AL_20260929): one per protected
        # tier per constraint type - 4 independent values, never merged
        self.lam_z = np.zeros(self.n_prot)
        self.lam_y = np.zeros(self.n_prot)
        self.last_eta = np.zeros(self.L)
        self.last_tbar = np.zeros(self.L)
        self.last_present = np.zeros(self.L, dtype=bool)
        self.last_gz = np.zeros(self.n_prot)   # signed residuals (P1)
        self.last_gy = np.zeros(self.n_prot)

        # per-tier price gains (fixed constants or adaptive, see step())
        self.eps_mode = c.eps_mode
        self.eps_z_vec = np.full(self.n_prot, float(np.clip(c.eps_z, c.eps_min, c.eps_max)))
        self.eps_y_vec = np.full(self.n_prot, float(np.clip(c.eps_y, c.eps_min, c.eps_max)))

        # mechanism layer: edge reservation pool for the top tier
        # alpha must stay below 1: at alpha = 1 the low pool's guaranteed
        # rate is zero (its tasks could wait forever)
        if c.reserve_alpha >= 1.0:
            raise ValueError(
                f"reserve_alpha must be < 1; got {c.reserve_alpha} "
                "(a full reservation leaves the low pool unserved)")
        if c.reserve_alpha > 0.0:
            self.reserve_alpha = float(c.reserve_alpha)
        else:
            self.reserve_alpha = min(0.5, 1.2 * float(self.tier_probs[self.L - 1]))
        self.reserve_cap = self.reserve_alpha * c.edge_freq * c.delta  # Gcycles/slot

        # capacity-normalized physical Lyapunov (review ruling): the reward
        # carries the ONE-STEP EXACT drift lambda_phy * (L(t+1) - L(t)),
        # which penalizes overload the moment queues build (the raw Q(A-mu)
        # form is silent at Q=0 and misses the new-backlog quadratic term)
        self.cap_loc = self.comp_freqs * c.delta                     # per device
        self.cap_high = self.reserve_alpha * c.edge_freq * c.delta
        self.cap_low = (1.0 - self.reserve_alpha) * c.edge_freq * c.delta

        # utility thresholds a_p: decreasing in tier (higher tier => narrower window)
        if self.L > 1:
            self.u_thresh = np.asarray(
                [c.u_a_base - c.u_a_slope * p / (self.L - 1) for p in range(self.L)]
            )
        else:
            self.u_thresh = np.asarray([c.u_a_base])

        # normalization references
        self.d_max = self.d_hi
        self.c_max = self.c_hi
        self.tau_max = self.d_hi * 1e6 * self.c_hi / c.std_freq
        self.q_scale = 10.0   # Gcycles
        self.z_scale = c.z_ref
        self.y_scale = c.y_ref

    # ------------------------------------------------------------------ #
    @property
    def edge_q(self):
        """Total edge backlog (both pools), for logging/observation."""
        return self.Qe_high + self.Qe_low

    def tx_rate_mbps(self, i, power_ratio):
        """Shannon rate in Mb/s for device i at a power ratio in [0,1].

        SNR = P*hi/(N0*B) in SI units; single source of truth for the
        wireless model (env.step, greedy baseline, observations).
        """
        p_w = self.trans_powers_w[i] * float(np.clip(power_ratio, 0.0, 1.0))
        snr = p_w * self.channel[i] / self.noise_watt
        if snr <= 0.0:
            return 0.0
        return self.bandwidth_hz * np.log2(1.0 + snr) / 1e6

    def _sample_tasks(self):
        c = self.cfg
        d = self.rng.uniform(self.d_lo, self.d_hi, (self.M, self.K))
        cc = self.rng.uniform(self.c_lo, self.c_hi, (self.M, self.K))
        tiers = self.rng.choice(self.L, (self.M, self.K), p=self.tier_probs)
        tau = d * 1e6 * cc / c.std_freq
        return d, cc, tau, tiers

    def reset(self):
        c = self.cfg
        self.local_q[:] = 0.0
        self.Qe_high = 0.0
        self.Qe_low = 0.0
        self.channel = self.path_loss * self.rng.exponential(1.0, self.M)
        self.task_d, self.task_c, self.task_tau, self.task_tier = self._sample_tasks()
        self.Z[:] = 0.0
        self.Y[:] = 0.0
        self.lam_z[:] = 0.0
        self.lam_y[:] = 0.0
        self.last_eta[:] = 0.0
        self.last_tbar[:] = 0.0
        self.last_present[:] = False
        self.last_gz[:] = 0.0
        self.last_gy[:] = 0.0
        self.eps_z_vec[:] = np.clip(c.eps_z, c.eps_min, c.eps_max)
        self.eps_y_vec[:] = np.clip(c.eps_y, c.eps_min, c.eps_max)
        return self._obs()

    def set_contract(self, eta_min, t_max):
        """Replace the protected-tier contract without changing dimensions."""
        eta = np.asarray(eta_min, dtype=float)
        delay = np.asarray(t_max, dtype=float)
        if len(eta) != self.n_prot or len(delay) != self.n_prot:
            raise ValueError(
                f"contract must define {self.n_prot} protected tiers")
        self.eta_min = eta.copy()
        self.t_max = delay.copy()
        self.cfg.eta_min = eta.tolist()
        self.cfg.t_max = delay.tolist()

    def current_obs(self):
        """Observation of the CURRENT state (for rollout chunks that do not
        reset the environment)."""
        return self._obs()

    def state_dict(self):
        """Full physical/price state for training resume (review ruling:
        resume must restore physical queues, Z/Y and the env RNG)."""
        return {
            "channel": self.channel.copy(),
            "task_d": self.task_d.copy(),
            "task_c": self.task_c.copy(),
            "task_tau": self.task_tau.copy(),
            "task_tier": self.task_tier.copy(),
            "local_q": self.local_q.copy(),
            "Qe_high": self.Qe_high,
            "Qe_low": self.Qe_low,
            "Z": self.Z.copy(),
            "Y": self.Y.copy(),
            "lam_z": self.lam_z.copy(),
            "lam_y": self.lam_y.copy(),
            "lam_at_boundary": bool(
                np.any(self.lam_z <= 1e-9) or np.any(self.lam_y <= 1e-9)
                or np.any(self.lam_z >= self.cfg.lambda_max - 1e-9)
                or np.any(self.lam_y >= self.cfg.lambda_max - 1e-9)),
            "eps_z_vec": self.eps_z_vec.copy(),
            "eps_y_vec": self.eps_y_vec.copy(),
            "last_eta": self.last_eta.copy(),
            "last_tbar": self.last_tbar.copy(),
            "last_present": self.last_present.copy(),
            "last_gz": self.last_gz.copy(),
            "last_gy": self.last_gy.copy(),
            "rng_bitgen": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self.channel = np.asarray(state["channel"], dtype=float).copy()
        self.task_d = np.asarray(state["task_d"], dtype=float).copy()
        self.task_c = np.asarray(state["task_c"], dtype=float).copy()
        self.task_tau = np.asarray(state["task_tau"], dtype=float).copy()
        self.task_tier = np.asarray(state["task_tier"], dtype=int).copy()
        self.local_q = np.asarray(state["local_q"], dtype=float).copy()
        self.Qe_high = float(state["Qe_high"])
        self.Qe_low = float(state["Qe_low"])
        self.Z = np.asarray(state["Z"], dtype=float).copy()
        self.Y = np.asarray(state["Y"], dtype=float).copy()
        self.lam_z = np.asarray(state["lam_z"], dtype=float).copy()
        self.lam_y = np.asarray(state["lam_y"], dtype=float).copy()
        self.eps_z_vec = np.asarray(state["eps_z_vec"], dtype=float).copy()
        self.eps_y_vec = np.asarray(state["eps_y_vec"], dtype=float).copy()
        self.last_eta = np.asarray(state["last_eta"], dtype=float).copy()
        self.last_tbar = np.asarray(state["last_tbar"], dtype=float).copy()
        self.last_present = np.asarray(state["last_present"], dtype=bool).copy()
        self.last_gz = np.asarray(state["last_gz"], dtype=float).copy()
        self.last_gy = np.asarray(state["last_gy"], dtype=float).copy()
        self.rng.bit_generator.state = state["rng_bitgen"]

    def physical_lyapunov(self, local_q, qe_high, qe_low):
        """Capacity-normalized physical Lyapunov:
        L = 1/2 [ sum_i (Qloc_i/C_loc_i)^2 + (Qh/C_h)^2 + (Ql/C_l)^2 ]."""
        x = float(np.sum((np.asarray(local_q) / self.cap_loc) ** 2))
        x += (qe_high / self.cap_high) ** 2
        x += (qe_low / self.cap_low) ** 2
        return 0.5 * x

    # ------------------------------------------------------------------ #
    def _age_priority(self, tier):
        """Class-level aged priority a_p(t) = p + beta_p(t) (opening Eq.(8)).

        The top tier never ages: structural priority and the reservation
        boundary are untouched. beta depends only on violation history, so
        the aging rate is decoupled from the nominal tier label.
        """
        c = self.cfg
        if tier >= self.L - 1 or c.aging_mode == "none":
            return float(tier)
        if c.aging_mode == "constant":
            return float(tier) + c.beta_0
        # aging_mode == "vq": price-driven slope
        if tier < self.n_prot:
            beta = c.beta_0 * (
                1.0 + c.kappa_a
                * (self.Z[tier] / self.z_scale + self.Y[tier] / self.y_scale)
            )
        else:
            beta = c.beta_0  # low but unprotected tier: no price available
        return float(tier) + beta

    def _task_order_key(self, i, k, tie_key):
        tier = int(self.task_tier[i, k])
        top_first = 0 if tier == self.L - 1 else 1
        return (top_first, -self._age_priority(tier), tie_key)

    def _obs(self):
        c = self.cfg
        # normalized log-rate: log2(1+SNR)/log2(1+SNR_ref) in (0, ~1]
        snr_hat = np.asarray(
            [np.log2(1.0 + p * h / self.noise_watt) / np.log2(1.0 + self.snr_ref)
             for p, h in zip(self.trans_powers_w, self.channel)],
            dtype=float,
        )
        # dual-pool backlogs: the aggregate edge_q is NOT a decision state
        # (same total, different pools => different transitions)
        parts = [np.array([self.Qe_high / self.q_scale, self.Qe_low / self.q_scale])]
        parts.append(self.local_q / self.q_scale)
        parts.append(snr_hat)
        # per-task features
        d_hat = (self.task_d / self.d_max).ravel()
        c_hat = (self.task_c / self.c_max).ravel()
        t_hat = (self.task_tau / self.tau_max).ravel()
        parts.append(d_hat)
        parts.append(c_hat)
        parts.append(t_hat)
        onehot = np.zeros((self.M * self.K, self.L))
        onehot[np.arange(self.M * self.K), self.task_tier.ravel()] = 1.0
        parts.append(onehot.ravel())
        # per-tier [Z, Y, eta, tbar]; absent tiers report (0, 0), which no
        # active tier can produce (an active tier always has tbar > 0).
        # Price squash: log1p (review rule - the linear Z/(Z+z_ref) squash
        # saturated at 1.0 during price-ramp runs); the 4*z_ref point maps
        # to 1.0, larger prices grow logarithmically without bound
        z = np.log1p(self.Z / self.z_scale) / np.log1p(4.0)
        y = np.log1p(self.Y / self.y_scale) / np.log1p(4.0)
        parts.append(z)
        parts.append(y)
        parts.append(self.last_eta)
        parts.append(np.clip(self.last_tbar, 0, 2) / 2)
        # P1 contract block (protected tiers): floors, ceilings, presence
        # mask and signed residuals - the policy can interpret the prices'
        # semantics, so a changed contract stays decodable (E8 basis)
        parts.append(self.eta_min.copy())
        parts.append(np.clip(self.t_max / 2.0, 0.0, 1.0))
        parts.append(self.last_present[: self.n_prot].astype(float))
        parts.append(np.clip(self.last_gz, -1.0, 1.0))
        parts.append(np.clip(self.last_gy / 2.0, -1.0, 4.0))
        if self.eps_mode == "adaptive":
            # gains shape the next queue update: observable => Markov prices
            parts.append(self.eps_z_vec / c.eps_max)
            parts.append(self.eps_y_vec / c.eps_max)
        if c.reward_mode == "al":
            # multipliers shape the reward: observable => Markov (freeze doc 4)
            parts.append(self.lam_z / c.lambda_max)
            parts.append(self.lam_y / c.lambda_max)
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == c.obs_dim, (obs.shape, c.obs_dim)
        return obs

    # ------------------------------------------------------------------ #
    def _decode_action(self, action):
        """action: flat [M*(K+1)] in [0,1] -> per-device offload matrix [M,K], power [M]."""
        a = np.asarray(action, dtype=float).reshape(self.M, self.K + 1)
        return np.clip(a[:, :self.K], 0, 1), np.clip(a[:, self.K], 0, 1)

    def step(self, action):
        c = self.cfg
        offl, pratio = self._decode_action(action)
        M, K, L = self.M, self.K, self.L
        top = L - 1

        trans_time = np.zeros((M, K))
        e_comp = np.zeros((M, K))
        dly = np.zeros((M, K))
        got = np.zeros((M, K))          # Mb actually transmitted (= D_edge)
        d_local = np.zeros((M, K))      # Mb kept local (= D - D_edge)
        e_tx_task = np.zeros((M, K))    # J
        energy_local_total = 0.0
        energy_tx_total = 0.0

        # aging priorities use the prices the agent observed (pre-update Z/Y)
        age = {p: self._age_priority(p) for p in range(L)}

        # ledgers: q_pre + arrivals == served + q_post (machine precision)
        led_local = np.zeros((M, 4))    # q_pre, arr, served, q_post per device
        led_high = np.zeros(4)
        led_low = np.zeros(4)
        led_local[:, 0] = self.local_q
        led_high[0] = self.Qe_high
        led_low[0] = self.Qe_low

        all_off = []                    # (i, k, got) with got > 0

        # ---- transmission stage: budget = rate*delta, D_edge = min(rho D, budget share)
        # NOTE: trans_time no longer enters the edge delay (data joins the
        # edge queue at slot end regardless); it still attributes E_tx.
        for i in range(M):
            rate_i = self.tx_rate_mbps(i, pratio[i])
            budget = max(0.0, rate_i * c.delta)
            order = sorted(
                range(K),
                key=lambda k: self._task_order_key(i, k, -offl[i, k] * self.task_d[i, k]),
            )
            cum = 0.0
            for k in order:
                want = offl[i, k] * self.task_d[i, k]
                g = min(want, budget)
                budget -= g
                got[i, k] = g
                d_local[i, k] = self.task_d[i, k] - g
                cum += g
                if g > 1e-12 and rate_i > 1e-12:
                    trans_time[i, k] = cum / rate_i
                    # E_tx = P * T_tx, attributed by each task's tx duration
                    e_tx_task[i, k] = self.trans_powers_w[i] * pratio[i] * (g / rate_i)
                    energy_tx_total += e_tx_task[i, k]
                    all_off.append((i, k, g))

            # ---- local computation (same device): Lindley with real arrivals
            total = self.local_q[i]
            lorder = sorted(
                range(K),
                key=lambda k: self._task_order_key(i, k, -d_local[i, k] * 1e6 * self.task_c[i, k]),
            )
            for k in lorder:
                w_l = d_local[i, k] * 1e6 * self.task_c[i, k]
                total += w_l
                led_local[i, 1] += w_l
                if w_l <= 1e-12:
                    dly[i, k] = 0.0  # no local work: edge branch governs
                else:
                    dly[i, k] = total / self.comp_freqs[i]
            led_local[i, 2] = min(total, self.comp_freqs[i] * c.delta)
            self.local_q[i] = max(0.0, total - self.comp_freqs[i] * c.delta)
            led_local[i, 3] = self.local_q[i]
            # COST/ENERGY SEMANTICS (review item 4): task-final basis - a
            # task's full local-processing energy is charged at its arrival
            # slot, exactly like the cost term; the per-slot physically
            # served energy stays available in ledger["local"][:, 2] (E0).
            energy_local_total += led_local[i, 1] * c.energy_coef
            self.channel[i] = self.path_loss[i] * self.rng.exponential(1.0)

        # ---- edge execution: per-pool Lindley + reservation + aging order
        new_high = []
        new_low = []
        for (i, k, g) in all_off:
            w_e = g * 1e6 * self.task_c[i, k]
            if int(self.task_tier[i, k]) == top:
                new_high.append((w_e, i, k))
            else:
                new_low.append((w_e, i, k))
        new_high.sort(key=lambda x: x[1])
        new_low.sort(key=lambda x: (-age[int(self.task_tier[x[1], x[2]])], x[1], x[2]))

        led_high[1] = sum(w for w, _, _ in new_high)
        led_low[1] = sum(w for w, _, _ in new_low)

        cap_e = c.edge_freq * c.delta
        # CAUSALITY (review blocker #1): service during slot t draws ONLY
        # from the backlog present at slot start; same-slot transmissions
        # join the pool at END of slot and are served from the next slot.
        high_served = min(led_high[0], self.reserve_cap)
        led_high[2] = high_served
        self.Qe_high = led_high[0] + led_high[1] - high_served
        led_high[3] = self.Qe_high

        low_budget = cap_e - high_served          # unused reservation flows down
        low_served = min(led_low[0], low_budget)
        led_low[2] = low_served
        self.Qe_low = led_low[0] + led_low[1] - low_served
        led_low[3] = self.Qe_low

        # Departure estimate for same-slot arrivals: they wait for the
        # slot-t residual of the old backlog, then drain at the pool's
        # GUARANTEED rate mu = (reservation share) * fe - the same formula
        # the greedy baseline uses (review item 2). The actual serve step
        # above may drain faster via the returned reservation; the estimate
        # is deliberately conservative and matches the theory's rates.
        mu_high = self.reserve_alpha * c.edge_freq
        mu_low = (1.0 - self.reserve_alpha) * c.edge_freq
        if mu_high > 0.0:
            # normal departure estimate
            base_high = c.delta + (led_high[0] - high_served) / mu_high
            for (w_e, i, k) in new_high:
                base_high += w_e / mu_high
                e_comp[i, k] = base_high
        elif led_high[0] + led_high[1] > 0.0:
            # zero reservation share but high-pool work exists: no service
            # path (infeasible configuration - prices diverge, E9 hook)
            for (w_e, i, k) in new_high:
                e_comp[i, k] = float("inf")
        # else: empty and inactive high pool - nothing to estimate
        # low pool: guaranteed rate is positive (reserve_alpha < 1 enforced)
        base_low = c.delta + (led_low[0] - low_served) / mu_low
        for (w_e, i, k) in new_low:
            base_low += w_e / mu_low
            e_comp[i, k] = base_low

        # ---- end-to-end delay, success, cost ----
        succ = np.zeros((M, K))
        tbar_task = np.zeros((M, K))
        e_branch = np.zeros((M, K))     # edge-branch delay (0 if not offloaded)
        cost = 0.0
        for i in range(M):
            for k in range(K):
                d = max(dly[i, k], e_comp[i, k])
                dly[i, k] = d
                tau = self.task_tau[i, k]
                tbar_task[i, k] = d / tau
                e_branch[i, k] = e_comp[i, k] / tau if e_comp[i, k] > 0 else 0.0
                succ[i, k] = 1.0 if d <= tau else 0.0
                # normalized cost: (local compute + tx) energy and edge price,
                # each against the full-task reference
                w_total = self.task_d[i, k] * 1e6 * self.task_c[i, k]
                e_task = c.energy_coef * d_local[i, k] * 1e6 * self.task_c[i, k] + e_tx_task[i, k]
                e_ref = c.energy_coef * w_total
                expn = c.service_price * got[i, k] * 1e6 * self.task_c[i, k]
                expn_ref = c.service_price * w_total
                cost += (c.energy_weight * min(e_task / max(e_ref, 1e-12), 1.0)
                         + c.expense_weight * min(expn / max(expn_ref, 1e-12), 1.0))

        # ---- per-tier slot statistics (current tasks, before resampling) ----
        eta = np.zeros(L)
        tbar = np.zeros(L)
        starv = np.zeros(L)
        util = np.zeros(L)
        present = np.zeros(L, dtype=bool)
        for p in range(L):
            mask = (self.task_tier == p)
            n = int(mask.sum())
            if n > 0:
                present[p] = True
                eta[p] = succ[mask].mean()
                tbar[p] = tbar_task[mask].mean()
                util[p] = float(utility_value(tbar_task[mask], self.u_thresh[p]).mean())
                if p < self.n_prot:
                    starv[p] = float((tbar_task[mask] > self.t_max[p]).mean())
        self.last_eta, self.last_tbar, self.last_present = eta, tbar, present

        # ---- virtual queues (protected low tiers, absent tiers skipped) ----
        # P1: SIGNED residuals - g < 0 when the tier over-satisfies, so the
        # price can fall back down (a ratchet is not a price)
        gz = np.zeros(self.n_prot)
        gy = np.zeros(self.n_prot)
        for p in range(self.n_prot):
            if not present[p]:
                continue  # explicit N_p = 0 masking: no deficit, no update
            gz[p] = self.eta_min[p] - eta[p]
            gy[p] = tbar[p] - self.t_max[p]
        self.last_gz, self.last_gy = gz.copy(), gy.copy()

        # reward uses the prices the agent OBSERVED (pre-update Z/Y);
        # "fixed" ablation keeps positive-part penalties (traditional form)
        if c.reward_mode == "vq":
            penalty = float(np.dot(self.Z, gz) + np.dot(self.Y, gy))
        elif c.reward_mode == "fixed":
            penalty = c.fixed_penalty * float(
                np.maximum(gz, 0.0).sum() + np.maximum(gy, 0.0).sum())
        elif c.reward_mode == "al":
            # projected augmented Lagrangian (DESIGN_FREEZE_AL_20260929):
            # normalized signed residuals, 4 independent multipliers,
            # pre-update lambda prices the reward, absent tiers skipped
            gz_n = np.zeros(self.n_prot)
            gy_n = np.zeros(self.n_prot)
            for p in range(self.n_prot):
                if not present[p]:
                    continue
                gz_n[p] = np.clip(gz[p], -1.0, 1.0)   # eta in [0,1]: bounded
                if c.delay_metric == "censored":
                    # per-task y = min(d/tau, 1); ceiling needs a (0,1)
                    # recalibration in the full route (vacuous under
                    # t_max > 1 - documented in the freeze doc 4)
                    yc = float(np.mean(np.minimum(
                        tbar_task[self.task_tier[:, :] == p] /
                        np.maximum(self.task_tau[self.task_tier == p], 1e-9), 1.0)))                         if np.any(self.task_tier == p) else 0.0
                    gy_n[p] = np.tanh(yc - self.t_max[p])
                else:  # "tanh": bounded relative overshoot, K0-K3 semantics
                    gy_n[p] = np.tanh((tbar[p] - self.t_max[p]) / self.t_max[p])
            pen = 0.0
            for p in range(self.n_prot):
                if not present[p]:
                    continue
                if c.al_rho > 0.0:   # augmented: ([lam + rho g]+^2 - lam^2)/(2 rho)
                    az = max(self.lam_z[p] + c.al_rho * gz_n[p], 0.0)
                    ay = max(self.lam_y[p] + c.al_rho * gy_n[p], 0.0)
                    pen += (az * az - self.lam_z[p] ** 2) / (2.0 * c.al_rho)
                    pen += (ay * ay - self.lam_y[p] ** 2) / (2.0 * c.al_rho)
                else:                # projected Lagrangian: lam * [g]+
                    pen += (self.lam_z[p] * max(gz_n[p], 0.0)
                            + self.lam_y[p] * max(gy_n[p], 0.0))
            penalty = float(pen)
        else:  # none
            penalty = 0.0

        for p in range(self.n_prot):
            if not present[p]:
                continue
            self.Z[p] = max(0.0, self.Z[p] + self.eps_z_vec[p] * gz[p])
            self.Y[p] = max(0.0, self.Y[p] + self.eps_y_vec[p] * gy[p])

        # AL multiplier update AFTER the reward (pre-update lambda prices
        # it): signed projected step, absent tiers skipped
        if c.reward_mode == "al":
            for p in range(self.n_prot):
                if not present[p]:
                    continue
                gz_n = np.clip(gz[p], -1.0, 1.0)
                gy_n = (np.tanh((tbar[p] - self.t_max[p]) / self.t_max[p])
                        if c.delay_metric == "tanh" else gz_n)
                self.lam_z[p] = float(np.clip(
                    self.lam_z[p] + c.alpha_lam * gz_n, 0.0, c.lambda_max))
                self.lam_y[p] = float(np.clip(
                    self.lam_y[p] + c.alpha_lam * gy_n, 0.0, c.lambda_max))

        # ---- reward (drift-plus-penalty + exact physical drift) ----
        # prices use the PRE-update Z/Y the agent observed; the physical
        # term is the exact one-step normalized Lyapunov drift
        # lambda_phy * (L(t+1) - L(t)) from the ACTUAL updated queues.
        # The drift is part of the SHARED reward foundation (cost +
        # physical stability) in every reward mode; the E7 ablation varies
        # the tier-price mechanism, not physical stability.
        l_pre = self.physical_lyapunov(led_local[:, 0], led_high[0], led_low[0])
        l_post = self.physical_lyapunov(led_local[:, 3], led_high[3], led_low[3])
        physical_drift = l_post - l_pre
        reward = -(c.V * cost + penalty + c.lambda_phy * physical_drift)

        # ---- adaptive gain update (produces NEXT slot's gains) ----
        if self.eps_mode == "adaptive":
            for p in range(self.n_prot):
                if not present[p]:
                    continue
                if gz[p] > 0.0:
                    self.eps_z_vec[p] = min(c.eps_max, self.eps_z_vec[p] * (1.0 + c.eta_g))
                else:
                    self.eps_z_vec[p] = max(c.eps_min, self.eps_z_vec[p] * (1.0 - c.eta_d))
                if gy[p] > 0.0:
                    self.eps_y_vec[p] = min(c.eps_max, self.eps_y_vec[p] * (1.0 + c.eta_g))
                else:
                    self.eps_y_vec[p] = max(c.eps_min, self.eps_y_vec[p] * (1.0 - c.eta_d))

        # tiers of the tasks just evaluated (snapshot before resampling)
        tiers_now = self.task_tier.copy()

        # ---- new tasks for next slot ----
        self.task_d, self.task_c, self.task_tau, self.task_tier = self._sample_tasks()

        info = {
            "cost": cost,
            # task-final basis, same accounting as "cost" (see comment above)
            "energy": energy_local_total,
            "tx_energy": energy_tx_total,
            "energy_total": energy_local_total + energy_tx_total,
            # physically served local energy this slot (E0/physics view)
            "energy_served_local": float(led_local[:, 2].sum() * c.energy_coef),
            "tx_bits": float(got.sum()),
            "edge_q": self.edge_q,
            # physical stability diagnostics (normalized queue levels and
            # the exact one-step Lyapunov drift charged in the reward)
            "physical_drift": float(physical_drift),
            "physical_drift_penalty": float(c.lambda_phy * physical_drift),
            "local_queue_norm": float(np.mean(np.asarray(led_local[:, 3]) / self.cap_loc)),
            "edge_high_queue_norm": float(led_high[3] / self.cap_high),
            "edge_low_queue_norm": float(led_low[3] / self.cap_low),
            "eta": eta.copy(),
            "tbar": tbar.copy(),
            "present": present.copy(),
            "utility": util.copy(),
            "starv": starv.copy(),
            "gz": gz,
            "gy": gy,
            "Z": self.Z.copy(),
            "Y": self.Y.copy(),
            "lam_z": self.lam_z.copy(),
            "lam_y": self.lam_y.copy(),
            "lam_at_boundary": bool(
                np.any(self.lam_z <= 1e-9) or np.any(self.lam_y <= 1e-9)
                or np.any(self.lam_z >= self.cfg.lambda_max - 1e-9)
                or np.any(self.lam_y >= self.cfg.lambda_max - 1e-9)),
            "eps_z": self.eps_z_vec.copy(),
            "eps_y": self.eps_y_vec.copy(),
            "tbar_task": tbar_task.copy(),
            "edge_branch_norm": e_branch.copy(),
            "edge_branch_delay": e_comp.copy(),   # absolute seconds (pre-resample tau)
            "delay": dly.copy(),
            "d_edge": got.copy(),          # Mb actually transmitted per task
            "d_local": d_local.copy(),     # Mb processed locally per task
            "tiers": tiers_now,
            # conservation ledger: columns = [q_pre, arrivals, served, q_post]
            "ledger": {
                "local": led_local.copy(),
                "edge_high": led_high.copy(),
                "edge_low": led_low.copy(),
            },
        }
        return self._obs(), reward, False, info

    # ------------------------------------------------------------------ #
    def episode_metrics(self, logs):
        """Aggregate per-slot info dicts into episode-level metrics.

        Absent tiers are excluded from per-tier averages (explicit N_p = 0
        masking) instead of being folded in as perfect slots.
        """
        L = self.L
        out = {
            "cost": float(np.mean([x["cost"] for x in logs])),
            "energy": float(np.mean([x["energy"] for x in logs])),
            "tx_energy": float(np.mean([x["tx_energy"] for x in logs])),
            "energy_total": float(np.mean([x["energy_total"] for x in logs])),
            "tx_bits": float(np.mean([x["tx_bits"] for x in logs])),
            "edge_q": float(np.mean([x["edge_q"] for x in logs])),
            # physical stability logs (review ruling)
            "physical_drift": float(np.mean([x["physical_drift"] for x in logs])),
            "physical_drift_penalty": float(np.mean([x["physical_drift_penalty"] for x in logs])),
            "local_queue_norm": float(np.mean([x["local_queue_norm"] for x in logs])),
            "edge_high_queue_norm": float(np.mean([x["edge_high_queue_norm"] for x in logs])),
            "edge_low_queue_norm": float(np.mean([x["edge_low_queue_norm"] for x in logs])),
            "tier_completion": np.zeros(L),
            "tier_norm_delay": np.zeros(L),
            "tier_max_norm_delay": np.zeros(L),
            "tier_starvation_rate": np.zeros(L),
            "tier_utility": np.zeros(L),
            "tier_p95_norm_delay": np.zeros(L),
            "tier_delay_samples": [[] for _ in range(L)],
            "tier_edge_delay_samples": [[] for _ in range(L)],
            "Z_final": logs[-1]["Z"].copy() if logs else np.zeros(self.n_prot),
            "Y_final": logs[-1]["Y"].copy() if logs else np.zeros(self.n_prot),
            # max compressed price observations over the chunk (log1p squash,
            # monotone in Z/Y; the 4*z_ref point maps to 1.0)
            "z_obs_max": (lambda m: (np.log1p(m / self.z_scale) / np.log1p(4.0)).tolist())(
                np.max(np.stack([x["Z"] for x in logs]), axis=0)
            ) if logs else [0.0] * self.n_prot,
            "y_obs_max": (lambda m: (np.log1p(m / self.y_scale) / np.log1p(4.0)).tolist())(
                np.max(np.stack([x["Y"] for x in logs]), axis=0)
            ) if logs else [0.0] * self.n_prot,
            # positive-part violation (signed residuals can go negative on
            # over-satisfaction; a violation metric must not)
            "violation": float(np.mean([
                np.maximum(x["gz"], 0.0).sum() + np.maximum(x["gy"], 0.0).sum()
                for x in logs
            ])),
        }
        eta_stack = np.stack([x["eta"] for x in logs])
        tbar_stack = np.stack([x["tbar"] for x in logs])
        util_stack = np.stack([x["utility"] for x in logs])
        starv_stack = np.stack([x["starv"] for x in logs])
        pres_stack = np.stack([x["present"] for x in logs]).astype(float)

        for x in logs:
            tiers = x["tiers"]
            for p in range(L):
                mask = (tiers == p)
                out["tier_delay_samples"][p].extend(x["tbar_task"][mask].tolist())
                e_br = x["edge_branch_norm"][mask]
                out["tier_edge_delay_samples"][p].extend(e_br[e_br > 0].tolist())

        for p in range(L):
            cnt = pres_stack[:, p].sum()
            if cnt > 0:
                out["tier_completion"][p] = float(
                    (eta_stack[:, p] * pres_stack[:, p]).sum() / cnt
                )
                out["tier_norm_delay"][p] = float(
                    (tbar_stack[:, p] * pres_stack[:, p]).sum() / cnt
                )
                out["tier_starvation_rate"][p] = float(
                    (starv_stack[:, p] * pres_stack[:, p]).sum() / cnt
                )
                out["tier_utility"][p] = float(
                    (util_stack[:, p] * pres_stack[:, p]).sum() / cnt
                )
                samples = np.asarray(out["tier_delay_samples"][p], dtype=float)
                if samples.size:
                    out["tier_max_norm_delay"][p] = float(samples.max())
                    out["tier_p95_norm_delay"][p] = float(np.percentile(samples, 95))
        return out
