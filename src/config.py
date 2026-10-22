"""Configuration for the virtual-queue-guided DRL offloading project."""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import List


def _str2bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class Params:
    # ----- run control -----
    run_name: str = "vq_default"
    seed: int = 1234
    eval_seed: int = 2345
    reward_mode: str = "vq"          # "vq" | "fixed" | "none" | "al"
    policy: str = "ppo"              # "ppo" | "random" | "static" | "greedy" | "edge_only"
    out_root: str = "results"

    # ----- scenario -----
    devices: int = 4
    tasks_per_slot: int = 2
    tiers: int = 3                   # L, larger = more critical
    protected_low: int = 2           # tiers 1..protected_low get soft guarantees
    time_slots: int = 200
    delta: float = 0.5
    total_bandwidth: float = 10e6
    trans_powers: List[float] = field(default_factory=lambda: [251.0, 206.0, 126.0, 227.0])
    path_loss: List[float] = field(default_factory=lambda: [2.4e-10, 7.6e-11, 8.0e-11, 1.0e-10])
    noise_dens_dbm: float = -174.0
    comp_freqs: List[float] = field(default_factory=lambda: [2.1, 2.5, 2.8, 2.3])
    # calibrated operating point (non-learning pre-pass 2026-09-28, P0
    # physics): edge_freq=24 -> low-pool utilization 0.94 with negative
    # drift; greedy/edge_only strongly differentiated (viol 0.77 vs 3.01).
    # Contracts = [0.85x, 1.15x] greedy natural levels at this point.
    edge_freq: float = 24.0
    std_freq: float = 2.0            # f0 in deadline definition
    energy_coef: float = 1.0         # kappa (J/Gcycle)
    service_price: float = 0.1       # $/Gcycle
    data_size_kb: List[float] = field(default_factory=lambda: [20.0, 200.0])
    comp_dens: List[float] = field(default_factory=lambda: [200.0, 2000.0])
    tier_probs: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    energy_weight: float = 0.8
    expense_weight: float = 0.2

    # ----- two-tier guarantees (tiers 1..protected_low) -----
    # calibrated at edge_freq=24 (greedy reference, 0.85x / 1.15x rule)
    eta_min: List[float] = field(default_factory=lambda: [0.38, 0.55])   # completion floors
    t_max: List[float] = field(default_factory=lambda: [2.0, 1.75])      # norm-delay ceilings
    eps_z: float = 0.05              # virtual-queue step (completion)
    eps_y: float = 0.05              # virtual-queue step (delay)
    eps_mode: str = "fixed"          # "fixed" | "adaptive" (per-tier self-calibration)
    eta_g: float = 0.03              # adaptive: multiplicative growth per violating slot
    eta_d: float = 0.03              # adaptive: multiplicative decay per met slot
    eps_min: float = 0.01            # adaptive: gain clip floor
    eps_max: float = 0.20            # adaptive: gain clip ceiling (T2 bound uses this)
    z_ref: float = 5.0               # virtual-queue scale (observation, aging, theory)
    y_ref: float = 5.0

    # ----- mechanism layer (opening-report alignment) -----
    reserve_alpha: float = 0.30      # top-tier edge budget share (1.2x top-tier prob)
    aging_mode: str = "vq"           # "vq" | "constant" | "none"
    beta_0: float = 0.2              # base aging slope (decoupled from nominal tier)
    kappa_a: float = 2.0             # price coupling of the aging slope (aging_mode="vq")

    # ----- task utility semantics (opening-report Eq.(4)) -----
    u_a_base: float = 0.65           # full-satisfaction threshold of the lowest tier
    u_a_slope: float = 0.15          # per-tier narrowing (higher tier => narrower window)

    # ----- reward -----
    V: float = 10.0                  # efficiency weight in drift-plus-penalty
    fixed_penalty: float = 5.0       # lambda_f for reward_mode="fixed"
    # P3 ruling: fixed proportional TRAINING reward scaling (no running
    # normalization, V untouched, no per-batch standardization):
    #   S = reward_scale_factor * max(1, V*M*K);  r_train = r_raw / S
    # The env keeps returning r_raw; only GAE and the critic see r_train.
    # A positive constant divisor leaves the optimal policy and the
    # cost/QoS weighting at the given V unchanged.
    # factor = 1/(1-gamma) = 100 matches the discounted effective horizon,
    # keeping long-run returns at unit-to-tens magnitude (review ruling).
    reward_scale_factor: float = 100.0
    reward_scale: float = 0.0        # derived in finalize(); actual S
    # exact one-step physical Lyapunov drift weight (review ruling):
    # r_raw = -[V*cost + prices + lambda_phy * (Lphy(t+1) - Lphy(t))]
    lambda_phy: float = 1.0
    # ----- augmented Lagrangian mode (DESIGN_FREEZE_AL_20260929) -----
    # reward_mode="al": penalty = sum_p AL(g, lam) with
    #   AL(g,lam) = ([lam + rho*g]+^2 - lam^2)/(2rho)   (rho>0)
    #   L(g,lam)  = lam*[g]+                             (rho=0, projected)
    # multipliers lam_z/lam_y per protected tier, projected to
    # [0, lambda_max], updated AFTER the reward with the signed residual
    # (pre-update lambda prices the reward). Global constants across
    # contracts - never per-contract tuned (probe gate 6).
    al_rho: float = 20.0             # augmentation coefficient (0 = projected)
    alpha_lam: float = 0.05          # multiplier step
    lambda_max: float = 50.0         # multiplier projection ceiling
    delay_metric: str = "tanh"       # "tanh" (bounded, K0-K3 semantics) | "censored"
    # capacity-driven stable warm start (review ruling): gate prob and
    # conditional Beta mean such that the expected effective offload sits
    # above the local stability floor; derived in finalize()
    rho_min: float = 0.0             # local stability floor
    rho_init: float = 0.0            # warm-start effective offload
    gate_init: float = 0.7           # initial gate-open probability
    # conditional Beta mean at init = rho_init / gate_init (~0.84):
    # a weaker open-gate prior now that physical drift anchors stability

    # ----- ppo -----
    episodes: int = 2000
    # P3 ruling (learning acceleration): update every 2 chunks (200 slots) -
    # the policy reacts to price changes twice as often; per-sample epoch
    # usage (p_epochs) is unchanged
    update_every: int = 2
    batch_size: int = 512
    p_epochs: int = 4
    v_epochs: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4        # separate critic optimizer (review ruling)
    clip: float = 0.2
    entropy: float = 0.01
    actor_grad_clip: float = 2.0
    critic_grad_clip: float = 10.0  # critic clip relaxed (review ruling)
    # > 0 enables per-epoch KL early stop (escalation option: 0.015)
    target_kl: float = 0.0
    # periodic checkpoint interval (episodes); must be a multiple of
    # update_every so the buffer is flushed at the save point; 0 disables
    checkpoint_every: int = 100
    hidden: int = 128

    # ----- evaluation -----
    eval_episodes: int = 100
    # evaluation policy mode (review ruling): "stochastic" is the MAIN
    # evaluation (long-term constrained problems allow randomized policies;
    # the Bernoulli gate is part of the final policy); "deterministic"
    # (thresholded gates, Beta means) is reported separately as a
    # deployment indicator and is NOT an E7 blocker
    policy_mode: str = "stochastic"  # "stochastic" | "deterministic"
    # steady-state burn-in: from the Z/Y = 0 cold start, run unlogged until
    # prices and physical queues are quasi-stable, then log statistics
    min_burn_slots: int = 5000
    burn_window: int = 2000          # stability check window
    max_burn_slots: int = 30000      # beyond this: report non-stationary
    burn_in_slots: int = 100         # unlogged warm-up before steady-state stats

    # ----- derived (filled by finalize) -----
    obs_dim: int = 0
    act_dim: int = 0

    def finalize(self):
        self.noise_power = (10 ** (self.noise_dens_dbm / 10.0)) * 1e-3  # mW per Hz
        self.data_size_mb = self.data_size_kb[1] * 1024 * 8 * 1e-6
        self.comp_dens_gc = self.comp_dens[1] * 1e-9
        self.protected_low = max(0, min(self.protected_low, self.tiers - 1))
        self.bandwidth = self.total_bandwidth / self.devices
        # actual training reward divisor S (written to config.json)
        if self.reward_scale_factor <= 0.0:
            raise ValueError("reward_scale_factor must be positive")
        self.reward_scale = self.reward_scale_factor * max(
            1.0, self.V * self.devices * self.tasks_per_slot)
        # capacity-driven warm start: rho_min = 1 - cap_local / E[workload]
        cap_local = sum(self.comp_freqs[:self.devices]) * self.delta
        d_mean = 0.5 * (self.data_size_kb[0] + self.data_size_kb[1]) * 1024 * 8 * 1e-6
        c_mean = 0.5 * (self.comp_dens[0] + self.comp_dens[1]) * 1e-9
        workload = self.devices * self.tasks_per_slot * d_mean * 1e6 * c_mean
        self.rho_min = max(0.0, 1.0 - cap_local / workload)
        self.rho_init = min(0.9, self.rho_min + 0.2)
        # obs: edge pools(2: Qe_high, Qe_low) + per-device(2M)
        #      + per-task((3+L)*M*K) + protected queues(2*n_prot: Z,Y)
        #      + tier stats(2*L: eta, tbar)
        #      + P1 contract block(5*n_prot: eta_min, t_max, present, gz, gy)
        #      + adaptive gains(2*n_prot: eps_z, eps_y per protected tier)
        #      + AL multipliers(2*n_prot: lam_z, lam_y per protected tier)
        self.obs_dim = 2 + 2 * self.devices \
            + (3 + self.tiers) * self.devices * self.tasks_per_slot \
            + 2 * self.protected_low + 2 * self.tiers \
            + 5 * self.protected_low \
            + (2 * self.protected_low if self.eps_mode == "adaptive" else 0) \
            + (2 * self.protected_low if self.reward_mode == "al" else 0)
        # latent action layout (zero-inflated Beta, review plan (d)):
        # [gates(MK), conditional rho(MK), power(M)]; the env receives
        # eff_rho = gate*rho and power zeroed for all-local devices
        self.act_dim = self.devices * (2 * self.tasks_per_slot + 1)
        return self

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d).finalize()


def build_parser():
    p = argparse.ArgumentParser(description="VQ-guided DRL offloading")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval_seed", type=int, default=None)
    p.add_argument("--reward_mode", type=str, choices=["vq", "fixed", "none", "al"], default=None)
    p.add_argument("--al_rho", type=float, default=None)
    p.add_argument("--alpha_lam", type=float, default=None)
    p.add_argument("--lambda_max", type=float, default=None)
    p.add_argument("--delay_metric", type=str, choices=["tanh", "censored"], default=None)
    p.add_argument("--policy", type=str,
                   choices=["ppo", "random", "static", "greedy", "edge_only"], default=None)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--tasks_per_slot", type=int, default=None)
    p.add_argument("--tiers", type=int, default=None)
    p.add_argument("--protected_low", type=int, default=None)
    p.add_argument("--time_slots", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--update_every", type=int, default=None)
    p.add_argument("--eval_episodes", type=int, default=None)
    p.add_argument("--policy_mode", type=str,
                   choices=["stochastic", "deterministic"], default=None)
    p.add_argument("--min_burn_slots", type=int, default=None)
    p.add_argument("--burn_window", type=int, default=None)
    p.add_argument("--max_burn_slots", type=int, default=None)
    p.add_argument("--V", type=float, default=None)
    p.add_argument("--fixed_penalty", type=float, default=None)
    p.add_argument("--reward_scale_factor", type=float, default=None)
    p.add_argument("--eps_z", type=float, default=None)
    p.add_argument("--eps_y", type=float, default=None)
    p.add_argument("--eps_mode", type=str, choices=["fixed", "adaptive"], default=None)
    p.add_argument("--eta_g", type=float, default=None)
    p.add_argument("--eta_d", type=float, default=None)
    p.add_argument("--eps_min", type=float, default=None)
    p.add_argument("--eps_max", type=float, default=None)
    p.add_argument("--reserve_alpha", type=float, default=None)
    p.add_argument("--aging_mode", type=str, choices=["vq", "constant", "none"], default=None)
    p.add_argument("--beta_0", type=float, default=None)
    p.add_argument("--kappa_a", type=float, default=None)
    p.add_argument("--u_a_base", type=float, default=None)
    p.add_argument("--u_a_slope", type=float, default=None)
    p.add_argument("--tier_probs", type=str, default=None,
                   help="comma-separated, length = tiers")
    p.add_argument("--out_root", type=str, default=None)
    return p


def params_from_args(argv=None, defaults: Params = None):
    args = build_parser().parse_args(argv)
    cfg = (defaults or Params()).finalize()
    for f in dataclasses.fields(cfg):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if args.tier_probs:
        cfg.tier_probs = [float(x) for x in args.tier_probs.split(",")]
        s = sum(cfg.tier_probs)
        cfg.tier_probs = [x / s for x in cfg.tier_probs]
    return cfg.finalize()
