"""Single-agent PPO with a ZERO-INFLATED BETA policy (review plan (d)).

Action-space fix for the hard blocker: a plain Beta can never emit an
exact rho = 0, so tau < delta tasks were structurally infeasible for the
learned policy (greedy can choose exact 0 and succeeds; the Beta policy
always opened the >= delta edge branch and failed).

Latent action per slot (buffer layout, act_dim = M*(2K+1)):
    gates  b_ik  ~ Bernoulli(q_ik)          # MK   offload on/off
    rho    r_ik  ~ Beta(alpha, beta)        # MK   conditional offload ratio
    power  p_i   ~ Beta(pa, pb)             # M    device power ratio
Environment action (what env.step receives):
    eff_rho_ik = b_ik * r_ik                # exact 0 when the gate is closed
    power_i    = p_i * any_k(b_ik)          # exact 0 when a device is all-local
log pi = sum log Bern(b) + b * log Beta(r) + log Beta(p)
(the conditional Beta fraction drops out of a task's log-prob when b = 0)

Deterministic policy: b = 1[q >= 0.5], rho = Beta mean, env power zeroed
per the any-gate rule. The critic is an independent network. GAE cuts at
intermediate dones and bootstraps only at the final state of the stitched
trajectory; every PPO epoch visits all mini-batches; sample-weighted
diagnostics (k3 KL, clip fraction, entropy, value error, grad norms,
boundary ratio of the conditional Beta) are reported per update.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta

CONC_LB = 2.0   # lower bound of the Beta concentration (avoids Dirac collapse)
ACT_EPS = 1e-6  # Beta support guard: returned fraction == logged fraction


def ortho_init(layer, gain=1.0):
    for name, p in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(p, 0.0)
        elif "weight" in name:
            nn.init.orthogonal_(p, gain=gain)


class Actor(nn.Module):
    """Shared task-conditioned actor (review ruling).

    Every task builds a LOCAL input x_ik = [D_hat, C_hat, tau_hat,
    tier_onehot, local_queue_i, channel_i, Z_p, Y_p, eta_min_p, t_max_p,
    present_p, gz_p, gy_p]; a SHARED encoder and SHARED per-task heads
    (conditioned on the global edge-pool context) produce the gate logit
    and the conditional Beta parameters. Device power comes from the
    task-embedding mean pooled per device. There are NO per-slot biases:
    identical tasks produce identical parameters, and outputs are
    permutation-equivariant within a device.
    """

    def __init__(self, obs_dim, n_task, n_dev, L, n_prot, hidden=128):
        super().__init__()
        self.n_task, self.n_dev = n_task, n_dev
        self.K = n_task // n_dev
        self.L, self.n_prot = L, n_prot
        self.fx_dim = 12 + L          # per-task local feature width
        self.df_dim = 2               # per-device features (local_q, SNR)
        self.glob_dim = 2             # edge pools

        self.task_encoder = nn.Sequential(
            nn.Linear(self.fx_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.glob_dim, hidden), nn.Tanh(),
        )
        self.gate_head = nn.Linear(hidden + hidden, 1)
        self.rho_mu_head = nn.Linear(hidden + hidden, 1)
        self.rho_conc_head = nn.Linear(hidden + hidden, 1)
        self.pow_mu_head = nn.Linear(hidden + self.df_dim + hidden, 1)
        self.pow_conc_head = nn.Linear(hidden + self.df_dim + hidden, 1)
        ortho_init(self.task_encoder[0])
        ortho_init(self.task_encoder[2])
        ortho_init(self.context_encoder[0])
        ortho_init(self.gate_head, gain=0.01)
        ortho_init(self.rho_mu_head, gain=0.01)
        ortho_init(self.rho_conc_head, gain=1e-3)
        ortho_init(self.pow_mu_head, gain=0.01)
        ortho_init(self.pow_conc_head, gain=1e-3)

    def _split_obs(self, obs):
        """Flat observation -> (per_task (B,MK,fx), dev (B,M,df), glob (B,g))."""
        B = obs.shape[0]
        M, K, L, n = self.n_dev, self.K, self.L, self.n_prot
        MK = M * K
        i = 0
        pools = obs[:, i:i + 2]; i += 2
        loc = obs[:, i:i + M]; i += M
        snr = obs[:, i:i + M]; i += M
        d = obs[:, i:i + MK]; i += MK
        cc = obs[:, i:i + MK]; i += MK
        tau = obs[:, i:i + MK]; i += MK
        onehot = obs[:, i:i + MK * L].reshape(B, MK, L); i += MK * L
        Z = obs[:, i:i + n]; i += n
        Y = obs[:, i:i + n]; i += n
        i += L          # last eta (tier means, subsumed by the tier block)
        i += L          # last tbar
        eta_min = obs[:, i:i + n]; i += n
        t_max = obs[:, i:i + n]; i += n
        present = obs[:, i:i + n]; i += n
        gz = obs[:, i:i + n]; i += n
        gy = obs[:, i:i + n]; i += n

        # per-task tier index from the one-hot block (non-differentiable
        # index use only); unprotected tiers read zero prices/contracts
        tier = onehot.argmax(-1)                                  # (B, MK)
        prot = (tier < n).float()

        def per_task_tier_vec(vec):
            v = torch.gather(vec, 1, tier.clamp(0, n - 1))
            return v * prot

        loc_t = loc.reshape(B, M, 1).expand(B, M, K).reshape(B, MK, 1)
        snr_t = snr.reshape(B, M, 1).expand(B, M, K).reshape(B, MK, 1)
        eta_t = per_task_tier_vec(eta_min).unsqueeze(-1)
        tm_t = per_task_tier_vec(t_max).unsqueeze(-1)
        pr_t = torch.gather(present, 1, tier.clamp(0, n - 1)).float().unsqueeze(-1)
        gz_t = per_task_tier_vec(gz).unsqueeze(-1)
        gy_t = per_task_tier_vec(gy).unsqueeze(-1)
        Z_t = per_task_tier_vec(Z).unsqueeze(-1)
        Y_t = per_task_tier_vec(Y).unsqueeze(-1)

        per_task = torch.cat([
            d.reshape(B, MK, 1), cc.reshape(B, MK, 1), tau.reshape(B, MK, 1),
            onehot, loc_t, snr_t, Z_t, Y_t, eta_t, tm_t, pr_t, gz_t, gy_t,
        ], dim=-1)
        dev = torch.stack([loc, snr], dim=-1)                     # (B, M, 2)
        return per_task, dev, pools

    def forward(self, obs):
        B = obs.shape[0]
        per_task, dev, glob = self._split_obs(obs)
        MK = self.n_task
        e = self.task_encoder(per_task.reshape(-1, self.fx_dim))
        e = e.reshape(B, MK, -1)                                   # (B, MK, H)
        ctx = self.context_encoder(glob)                           # (B, H)
        ctx_e = ctx.unsqueeze(1).expand(B, MK, ctx.shape[-1])
        gate_logits = self.gate_head(
            torch.cat([e, ctx_e], dim=-1)).squeeze(-1)             # (B, MK)
        u = torch.sigmoid(self.rho_mu_head(torch.cat([e, ctx_e], dim=-1))).squeeze(-1)
        conc = CONC_LB + F.softplus(
            self.rho_conc_head(torch.cat([e, ctx_e], dim=-1))).squeeze(-1)
        alpha = 1.0 + u * conc
        beta = 1.0 + (1.0 - u) * conc
        # device power from the per-device mean task embedding
        e_dev = e.reshape(B, self.n_dev, self.K, -1).mean(dim=2)   # (B, M, H)
        pow_in = torch.cat([e_dev, dev, ctx.unsqueeze(1).expand(
            B, self.n_dev, ctx.shape[-1])], dim=-1)                # (B, M, ...)
        p_mu = torch.sigmoid(self.pow_mu_head(pow_in)).squeeze(-1)
        p_conc = CONC_LB + F.softplus(self.pow_conc_head(pow_in)).squeeze(-1)
        p_alpha = 1.0 + p_mu * p_conc
        p_beta = 1.0 + (1.0 - p_mu) * p_conc
        return gate_logits, alpha, beta, p_alpha, p_beta


class Critic(nn.Module):
    """Independent value network: shares nothing with the actor."""

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.v_head = nn.Linear(hidden, 1)
        ortho_init(self.body[0])
        ortho_init(self.body[2])
        ortho_init(self.v_head)

    def forward(self, obs):
        return self.v_head(self.body(obs))


def compute_gae(rewards, values, dones, gamma, lam):
    """GAE over a stitched trajectory; the recursion is CUT at any
    intermediate done (nonterminal factor zeroes bootstrap and carry)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values[:-1], dtype=np.float32)
    return adv, returns


class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_task = cfg.devices * cfg.tasks_per_slot
        self.n_dev = cfg.devices
        self.actor = Actor(cfg.obs_dim, self.n_task, self.n_dev,
                           cfg.tiers, cfg.protected_low, cfg.hidden)
        self.critic = Critic(cfg.obs_dim, cfg.hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self._warm_start()

    def _warm_start(self):
        """Capacity-driven stable initialization (review ruling): gate
        probability gate_init and conditional Beta mean rho_init/gate_init,
        so the expected effective offload rho_init sits ABOVE the local
        stability floor rho_min = 1 - cap_local / E[workload]. A warm start
        inside the feasible region - the objective is unchanged."""
        cfg = self.cfg
        m = min(0.99, cfg.rho_init / cfg.gate_init)
        # conditional Beta mean (1 + u*c)/(2 + c) = m: pick a concentration
        # large enough that u stays inside (0, 1)
        c0 = max(24.0, CONC_LB + 1.0)
        u0 = (m * (2.0 + c0) - 1.0) / c0
        u0 = min(max(u0, 1e-4), 1.0 - 1e-4)
        q_logit = float(np.log(cfg.gate_init / (1.0 - cfg.gate_init)))
        conc_bias = float(np.log(np.expm1(c0 - CONC_LB)))
        with torch.no_grad():
            # biases carry the warm-start values; the WEIGHTS keep a full
            # orthogonal init so the task embedding drives the outputs from
            # step one (a zeroed weight made the gate a constant - the very
            # slot-bias pathology this round fixes)
            ortho_init(self.actor.gate_head, gain=1.0)
            self.actor.gate_head.bias.fill_(q_logit)
            ortho_init(self.actor.rho_mu_head, gain=1.0)
            self.actor.rho_mu_head.bias.fill_(float(np.log(u0 / (1.0 - u0))))
            self.actor.rho_conc_head.weight.zero_()
            self.actor.rho_conc_head.bias.fill_(conc_bias)
        # power heads keep their small-gain init

    # ---------------- acting ---------------- #
    def act(self, obs, deterministic=False):
        """Return (env_action, logp, value, latent_action).

        env_action: flat [M*(K+1)] = [eff_rho_ik..., power_i*any_gate]
        latent_action: flat [M*(2K+1)] = [b_ik..., r_ik..., p_i] (buffer layout)
        """
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            gate_logits, alpha, beta, p_alpha, p_beta = self.actor(o)
            gate_dist = Bernoulli(logits=gate_logits)
            rho_dist = Beta(alpha, beta)
            pow_dist = Beta(p_alpha, p_beta)
            if deterministic:
                b = (gate_dist.probs >= 0.5).float()
                rho = rho_dist.mean
                power = pow_dist.mean
            else:
                b = gate_dist.sample()
                rho = rho_dist.sample()
                power = pow_dist.sample()
            b = b.clamp(0.0, 1.0)                       # exact 0/1
            rho = rho.clamp(ACT_EPS, 1.0 - ACT_EPS)
            power = power.clamp(ACT_EPS, 1.0 - ACT_EPS)
            M, K = self.n_dev, self.n_task // self.n_dev
            any_gate = (b.reshape(M, K).sum(-1) > 0).float()
            # the env ignores the power of all-local devices: mask its
            # log-prob by the gate state (no noise gradient from unused
            # power actions, review fix)
            logp = (gate_dist.log_prob(b).sum(-1)
                    + (b * rho_dist.log_prob(rho)).sum(-1)
                    + (any_gate * pow_dist.log_prob(power).reshape(M)).sum(-1))
            v = self.critic(o)

            eff = (b * rho).reshape(M, K)
            power_env = power.reshape(M) * any_gate      # exact 0 when all-local
            env_action = torch.cat([eff, power_env.unsqueeze(1)], dim=1).reshape(-1)
            latent = torch.cat([b.reshape(-1), rho.reshape(-1), power.reshape(-1)])
            return (env_action.squeeze(0).numpy(),
                    float(logp),
                    float(v),
                    latent.squeeze(0).numpy())

    def value(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return float(self.critic(o))

    # ---------------- learning ---------------- #
    @staticmethod
    def _split_latent(act_t, n_task):
        gates = act_t[:, :n_task]
        rho = act_t[:, n_task:2 * n_task]
        power = act_t[:, 2 * n_task:]
        return gates, rho, power

    def _evaluate(self, obs_t, act_t):
        """Joint log-prob and entropy of latent (gates, conditional rho,
        power). b=0 tasks contribute no Beta fraction."""
        gate_logits, alpha, beta, p_alpha, p_beta = self.actor(obs_t)
        gate_dist = Bernoulli(logits=gate_logits)
        rho_dist = Beta(alpha, beta)
        pow_dist = Beta(p_alpha, p_beta)
        gates, rho, power = self._split_latent(act_t, self.n_task)
        any_gate = (gates.reshape(-1, self.n_dev,
                                  self.n_task // self.n_dev).sum(-1) > 0.0).float()
        # all-local devices: the env ignores their power, so the power
        # fraction is masked out of the log-prob (entropy weighted by
        # P(any gate open)) - no noise gradient from unused actions
        logp = (gate_dist.log_prob(gates).sum(-1)
                + (gates * rho_dist.log_prob(rho.clamp(ACT_EPS, 1.0 - ACT_EPS))).sum(-1)
                + (any_gate * pow_dist.log_prob(
                    power.clamp(ACT_EPS, 1.0 - ACT_EPS))).sum(-1))
        ent = ((gate_dist.entropy()
                + gate_dist.probs * rho_dist.entropy()).sum(-1)
               + (any_gate * pow_dist.entropy()).sum(-1))
        return logp, ent

    def update(self, obs, acts, logp_old, rewards, dones, values):
        """Update on the stitched trajectory of `update_every` chunks.
        `acts` are LATENT actions [gates, conditional rho, power]."""
        c = self.cfg
        T = len(rewards)
        adv, returns = compute_gae(rewards, values, dones, c.gamma, c.lam)

        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-5),
                                dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        act_t = torch.as_tensor(np.asarray(acts), dtype=torch.float32)
        logp_old_t = torch.as_tensor(np.asarray(logp_old), dtype=torch.float32)
        values_t = torch.as_tensor(np.asarray(values[:-1]), dtype=torch.float32)

        n = T
        mb = min(c.batch_size, n)

        kl_acc, clip_acc, ent_acc, p_loss_acc, gn_policy = [], [], [], [], []
        p_weight = 0
        p_updates = 0
        kl_early_stop = False
        p_epochs_run = 0
        _, rho_t, _ = self._split_latent(act_t, self.n_task)
        boundary_frac = float(((rho_t < 0.05) | (rho_t > 0.95)).float().mean())
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
                p_loss = -(torch.min(s1, s2) + c.entropy * ent).mean()
                self.actor_opt.zero_grad()
                p_loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.actor.parameters(), c.actor_grad_clip)
                self.actor_opt.step()
                p_updates += 1
                with torch.no_grad():
                    kl = (torch.exp(logratio) - 1.0 - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > c.clip).float().mean()
                b = len(ids)
                p_weight += b
                epoch_kl += float(kl) * b
                epoch_w += b
                kl_acc.append(float(kl) * b)
                clip_acc.append(float(clipfrac) * b)
                ent_acc.append(float(ent.mean()) * b)
                p_loss_acc.append(float(p_loss) * b)
                gn_policy.append(float(gnorm) * b)
            p_epochs_run += 1
            if c.target_kl > 0.0 and epoch_w > 0 and epoch_kl / epoch_w > c.target_kl:
                kl_early_stop = True
                break

        v_loss_acc, ve_acc, gn_critic = [], [], []
        v_weight = 0
        v_updates = 0
        critic_clipped = 0
        for _ in range(c.v_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, mb):
                ids = perm[start:start + mb]
                v = self.critic(obs_t[ids]).squeeze(-1)
                v_loss = nn.functional.mse_loss(v, ret_t[ids])
                self.critic_opt.zero_grad()
                v_loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.critic.parameters(), c.critic_grad_clip)
                self.critic_opt.step()
                v_updates += 1
                if float(gnorm) > c.critic_grad_clip:
                    critic_clipped += 1
                with torch.no_grad():
                    ve = (v - ret_t[ids]).abs().mean()
                b = len(ids)
                v_weight += b
                v_loss_acc.append(float(v_loss) * b)
                ve_acc.append(float(ve) * b)
                gn_critic.append(float(gnorm) * b)

        ret_np = ret_t.numpy()
        mean_abs_return = float(np.mean(np.abs(ret_np)))
        with torch.no_grad():
            v0 = self.critic(obs_t).squeeze(-1).numpy()
        var_ret = float(np.var(ret_np))
        explained_variance = (
            float(1.0 - np.var(ret_np - v0) / var_ret) if var_ret > 1e-12 else 0.0
        )
        value_error = float(np.sum(ve_acc) / v_weight) if v_weight else 0.0
        normalized_value_mae = value_error / (mean_abs_return + 1e-6)

        stats = {
            "p_loss": float(np.sum(p_loss_acc) / p_weight) if p_weight else 0.0,
            "v_loss": float(np.sum(v_loss_acc) / v_weight) if v_weight else 0.0,
            "entropy": float(np.sum(ent_acc) / p_weight) if p_weight else 0.0,
            "approx_kl": float(np.sum(kl_acc) / p_weight) if p_weight else 0.0,
            "clipfrac": float(np.sum(clip_acc) / p_weight) if p_weight else 0.0,
            "value_error": value_error,
            "normalized_value_mae": normalized_value_mae,
            "explained_variance": explained_variance,
            "critic_clipfrac": critic_clipped / v_updates if v_updates else 0.0,
            "mean_abs_return": mean_abs_return,
            "actor_grad_norm": float(np.sum(gn_policy) / p_weight) if p_weight else 0.0,
            "critic_grad_norm": float(np.sum(gn_critic) / v_weight) if v_weight else 0.0,
            "boundary_frac": boundary_frac,
            "p_updates": p_updates,
            "v_updates": v_updates,
            "p_epochs_run": p_epochs_run,
            "kl_early_stop": kl_early_stop,
            "target_kl": float(c.target_kl),
            "p_weight": p_weight,
            "v_weight": v_weight,
            "reward_scale": float(c.reward_scale),
            "rollout_samples": T,
        }
        return stats

    def save(self, path, with_optimizers=False):
        payload = {"actor": self.actor.state_dict(),
                   "critic": self.critic.state_dict()}
        if with_optimizers:
            payload["actor_opt"] = self.actor_opt.state_dict()
            payload["critic_opt"] = self.critic_opt.state_dict()
        torch.save(payload, path)

    def load(self, path, with_optimizers=False):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        if with_optimizers and "actor_opt" in d:
            self.actor_opt.load_state_dict(d["actor_opt"])
            self.critic_opt.load_state_dict(d["critic_opt"])
