"""Audited theory helpers (THEORY_AUDIT_20260929.md, route 2).

Corrected statements for the non-learning per-slot drift-minimizing
controller:

  T1'/T2' (signed residuals, finite horizon, NO stationarity assumption):
      max(0, x) >= x  =>  Z(T) >= eps * sum_{t<T} g(t)
      =>  (1/T) sum g(t) <= Z(T)/(eps*T)
      =>  (1/T) sum eta(t) >= eta_min - Z(T)/(eps*T)
  The earlier "[0,g_max] + E[g] <= -delta" pair was self-contradictory and
  is withdrawn.

  T1b' (dual-pool, end-of-slot admission): quadratic drift bound with
  bounded increments; A-terms are one-step approximations (this slot's
  action decides NEXT slot's edge arrivals) - stated explicitly.

  T3' (xi-approximate DPP): one-step optimality holds for the controller
  with a MEASURED per-slot suboptimality xi; no PPO claim.

  T4': only under the CENSORED delay definition y = min(d/tau, 1)
  (bounded residual); the env switch to per-task censored Y increments is
  a pending code task for the full route.

  T5+: rate-form exponential tail with an UNSPECIFIED prefactor C_exp
  (the C=1 claim is withdrawn until the prefactor is derived for the
  dual-pool end-of-slot model).
"""

import argparse
import glob
import json
import os

import numpy as np


def finite_horizon_residual_bound(z_T, eps, T):
    """T1'/T2': (1/T) sum g(t) <= Z(T)/(eps*T) for signed residuals - no
    stationarity or feasibility assumption (max(0,x) >= x)."""
    if T <= 0 or eps <= 0:
        return float("inf")
    return z_T / (eps * T)


def finite_horizon_eta_bound(eta_min, z_T, eps, T):
    """T2' corollary: mean completion >= eta_min - Z(T)/(eps*T)."""
    return eta_min - finite_horizon_residual_bound(z_T, eps, T)


def censored_delay_bound(d_max, y_T, eps, T):
    """T4': mean CENSORED delay <= d_max + Y(T)/(eps*T)."""
    return d_max + finite_horizon_residual_bound(y_T, eps, T)


def physical_pool_bound(B, delta, q0=0.0):
    """Lemma 1b: E[Q_low] <= Q0 + B / delta (Gcycles)."""
    return q0 + B / delta


def wait_bound(q_bar, mu_low, w):
    """T5: P(W > w) <= E[Q_low] / (mu_low * w); values >= 1 are vacuous."""
    if w <= 0:
        return float("inf")
    return q_bar / (mu_low * w)


# --------------------------------------------------------------------- #
# T5+ (v2.0): exponential bounded-wait bound via Chernoff drift.
#
# Setting: slot work arriving to the low pool A^t = sum of at most M*K
# i.i.d. task works w = D*C (D uniform, C uniform, independent), so A is
# bounded and its MGF is the (MK)-th power of the task-work MGF. The low
# pool drains at mu_slot Gcycles/slot (conservatively the edge low pool
# only; device-local service only makes the true drift more negative).
# Let gamma* = sup{ gamma > 0 : MK*log M_w(gamma) - gamma*mu_slot < 0 }.
# The standard exponential-drift argument (Hajek; Neely Ch.6; Li-Neely)
# then gives the stationary tail-rate form
#     P(W_sec > w) <= C * exp( -gamma* * mu_slot * w / delta_t ),
# with delta_t the slot duration and C a bounded prefactor fixed by the
# drift lemma (the numeric table reports the rate-form C=1 bound and says
# so). gamma* exists and is positive whenever E[A] < mu_slot; it is
# computed here NUMERICALLY (2-D Gauss-Legendre for M_w), not asymptotically.
# --------------------------------------------------------------------- #

def _task_work_mgf(gamma, d_lo, d_hi, c_lo, c_hi, n_nodes=48):
    """MGF of one task's work w = D * 1e6 * C (Gcycles).

    D ~ U[d_lo, d_hi] Mb, C ~ U[c_lo, c_hi] Gcycles/bit, independent.
    Computed by nested Gauss-Legendre quadrature; the integrand
    exp(gamma * 1e6 * x * y) is smooth and positive on the box, so the
    rule converges geometrically.
    """
    xi, wi = np.polynomial.legendre.leggauss(n_nodes)
    x = 0.5 * (d_hi - d_lo) * xi + 0.5 * (d_hi + d_lo)
    wx = 0.5 * (d_hi - d_lo) * wi
    y = 0.5 * (c_hi - c_lo) * xi + 0.5 * (c_hi + c_lo)
    wy = 0.5 * (c_hi - c_lo) * wi

    total = 0.0
    for xv, wxv in zip(x, wx):
        t = gamma * 1e6 * xv
        if abs(t) * (c_hi - c_lo) < 1e-12:
            inner = np.dot(wy, np.ones_like(y))
        else:
            inner = np.dot(wy, np.exp(t * y))
        total += wxv * inner
    return total / ((d_hi - d_lo) * (c_hi - c_lo))


def chernoff_rate(devices, tasks_per_slot, mu_slot, d_lo, d_hi, c_lo, c_hi):
    """Solve gamma* = sup{gamma: MK*log M_w(gamma) < gamma*mu_slot}.

    Returns (gamma_star, diagnostics dict). gamma_star is None when the
    reference load already exceeds the service rate (no negative drift;
    the exponential bound is vacuous and T2 feasibility fails).
    """
    n = devices * tasks_per_slot
    e_work = 0.5 * (d_hi - d_lo) * 0.5 * (c_hi - c_lo) * 1e6 + \
        0.5 * (d_hi + d_lo) * 1e6 * 0.5 * (c_hi + c_lo)  # E[D]E[C] first order
    e_work = 0.5 * (d_lo + d_hi) * 1e6 * 0.5 * (c_lo + c_hi)  # exact for uniforms

    def f(g):
        if g <= 0:
            return -e_work * n  # f(0) = 0, f'(0) = E[A] - mu
        return n * np.log(_task_work_mgf(g, d_lo, d_hi, c_lo, c_hi)) - g * mu_slot

    if e_work * n >= mu_slot:
        # full offloading saturates the low pool; the negative-drift premise
        # still holds for any policy with pool load below mu_slot, i.e. an
        # average offload ratio <= max_offload_ratio (report it explicitly)
        return None, {"load": e_work * n, "mu_slot": mu_slot,
                      "utilization": e_work * n / mu_slot, "feasible": False,
                      "max_offload_ratio": mu_slot / (e_work * n)}

    # bracket the crossing: f is convex in gamma (log-MGF of a bounded var),
    # negative at 0+, crosses once at gamma*
    hi = 1e-6
    while f(hi) < 0 and hi < 1e3:
        hi *= 2.0
    if f(hi) < 0:  # no crossing below 1e3: effectively stable for all gamma
        return 1e3, {"load": e_work * n, "mu_slot": mu_slot,
                     "utilization": e_work * n / mu_slot, "feasible": True}
    lo = hi * 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    gamma_star = 0.5 * (lo + hi)
    return gamma_star, {"load": e_work * n, "mu_slot": mu_slot,
                        "utilization": e_work * n / mu_slot, "feasible": True}


def exp_wait_bound(gamma_star, mu_slot, delta_t, w_seconds):
    """T5+ RATE form with an UNSPECIFIED prefactor C_exp:

        P(W > w) <= C_exp * exp(-gamma* * mu_slot * w / delta_t)

    The earlier C=1 claim is WITHDRAWN (prefactor not yet derived for the
    dual-pool end-of-slot model); this function returns the exponent rate
    only - do not report it as a complete probability bound.
    """
    if gamma_star is None or w_seconds <= 0:
        return float("inf")
    return float(np.exp(-gamma_star * mu_slot * w_seconds / delta_t))


def _default_numbers():
    from src.config import Params

    c = Params().finalize()
    alpha = c.reserve_alpha if c.reserve_alpha > 0 else min(
        0.5, 1.2 * c.tier_probs[-1] / sum(c.tier_probs))
    mu_low = (1.0 - alpha) * c.edge_freq * c.delta  # Gcycles per slot
    d_lo = c.data_size_kb[0] * 1024 * 8 * 1e-6
    d_hi = c.data_size_kb[1] * 1024 * 8 * 1e-6
    c_lo = c.comp_dens[0] * 1e-9
    c_hi = c.comp_dens[1] * 1e-9
    tau_ref = 0.5 * (d_lo + d_hi) * 1e6 * 0.5 * (c_lo + c_hi) / c.std_freq
    return c, alpha, mu_low, tau_ref, d_lo, d_hi, c_lo, c_hi


def numeric_table():
    """Audited finite-horizon table (T1'/T2'/T4'): the bound value depends
    on the TERMINAL price Z(T) and the horizon T, so the table reports the
    residual/eta bounds for a grid of (Z_T, T) pairs a run may end with.
    The delta-based columns are withdrawn (contradictory assumptions)."""
    c, alpha, mu_low, tau_ref, d_lo, d_hi, c_lo, c_hi = _default_numbers()
    gamma_star, diag = chernoff_rate(c.devices, c.tasks_per_slot, mu_low,
                                     d_lo, d_hi, c_lo, c_hi)
    rows = []
    for z_T in [1.0, 10.0, 50.0, 200.0]:
        for T in [2_000, 10_000, 50_000]:
            row = {
                "Z_terminal": z_T,
                "horizon_T": T,
                "T1_residual_bound": finite_horizon_residual_bound(
                    z_T, c.eps_z, T),
                "T2_eta_floor_tier1": finite_horizon_eta_bound(
                    c.eta_min[0], z_T, c.eps_z, T),
                "T4_censored_delay_ceiling_tier1": censored_delay_bound(
                    c.t_max[0], z_T, c.eps_y, T),
            }
            rows.append(row)
    meta = {
        "gamma_star": gamma_star,
        "load_Gcycles_per_slot": diag["load"],
        "mu_low_Gcycles_per_slot": diag["mu_slot"],
        "edge_low_pool_utilization": diag["utilization"],
        "chernoff_rate_feasible": diag["feasible"],
        "tau_ref_seconds": tau_ref,
        "note": "T5+ exponent RATE only (prefactor C_exp unspecified, C=1 "
                "withdrawn); Markov column withdrawn pending dual-pool "
                "end-of-slot derivation.",
    }
    return rows, meta


def check_against_results():
    """Compare T1'/T2' finite-horizon identities against evaluation JSONs.

    Scope (final audit): T1'/T2' are path identities of the SAME system
    the experiments run, so they are checkable from recorded terminal
    prices Z(T) and the reported eta. T4' is NOT checked: the frozen
    experiments use the UNCENSORED delay metric while T4' is stated for
    y = min(d/tau, 1) - a different system. T5+ is printed as an exponent
    RATE only (prefactor unspecified - never a probability bound).
    """
    # scope: the FROZEN evidence families only. Historical debug runs
    # (gate300*, zib_*, phy_*, p3i_*, reconf_*) predate the continuous
    # evaluation protocol; their aggregated eta / chunk-final-Z proxies
    # mismatch on oscillating paths and produce spurious EXCEEDS that say
    # nothing about the identity itself.
    frozen = ["calibB_*", "e7p_*", "e7_phase2/*", "e8_transfer/*",
              "e8_frozen_multiseed/*", "dpp_probe/*", "calib_*",
              "lam*"]
    hits = []
    for pat in frozen:
        hits.extend(glob.glob(os.path.join("results", pat, "evaluation.json")))
    hits = sorted(set(hits))
    if not hits:
        print("[check] no frozen-evidence evaluation.json found")
        return
    rows, meta = numeric_table()
    gs = meta["gamma_star"]
    print(f"[check] {len(hits)} evaluation file(s)")
    print(f"[check] T5+ gamma* = {gs if gs is None else round(gs, 4)} "
          f"(exponent RATE only; prefactor C_exp unspecified)")
    from src.config import Params
    base_cfg = Params().finalize()
    for path in hits:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        eta = s.get("tier_completion")
        z_fin = s.get("Z_final")
        line = os.path.relpath(path)
        if eta and z_fin:
            # T2': mean eta >= eta_min - (Z(T)-Z(0))/(eps*T); Z(0)=0 in
            # these cold-started evaluation runs
            T = int(s.get("eval_episodes", 100)) * base_cfg.time_slots
            for p in range(min(2, len(eta))):
                bound = finite_horizon_eta_bound(
                    base_cfg.eta_min[p], float(z_fin[p]),
                    base_cfg.eps_z, T)
                flag = "OK" if eta[p] >= bound - 1e-9 else "EXCEEDS"
                print(f"  {line}: T2' tier{p + 1} eta={eta[p]:.4f} "
                      f">= eta_min - (Z(T)-Z(0))/(eps*T) = {bound:.4f} "
                      f"-> {flag}")
        if s.get("Y_final") is not None:
            print(f"  {line}: Y_final={np.round(s['Y_final'], 1).tolist()} "
                  f"(UNCENSORED metric - informational, not a T4' check)")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare bounds against results/*/evaluation.json")
    args = ap.parse_args()
    if args.check:
        check_against_results()
        print()
    rows, meta = numeric_table()
    print("T5+ metadata:", json.dumps(meta, indent=2))
    print(json.dumps(rows, indent=2))
    print("\nInterpretation (final contracted scope):")
    print(" - T1'/T2' are finite-horizon PATH identities (no")
    print("   stationarity assumption):")
    print("   (1/T) sum g(t) <= (Z(T)-Z(0))/(eps*T) over the tier's")
    print("   ACTIVE slots; Z(0)=0 only in cold-started runs.")
    print(" - The dual-pool one-step physical drift expansion is")
    print("   retained as a derivation (bounded increments;")
    print("   first-order arrival approximation stated).")
    print(" - T3' (xi-approximate DPP) and T4' (censored delay) are")
    print("   CONDITIONAL EXTENSIONS, not verified claims: T3' is a")
    print("   proof template (B_eff / coupling terms / V,xi scaling")
    print("   not fully assembled); T4' is stated for y=min(d/tau,1)")
    print("   which the frozen experiments do NOT use. Neither may")
    print("   appear in the abstract, contribution list or experiment")
    print("   conclusions.")
    print(" - T5+ is FUTURE WORK: the exponent rate gamma* is")
    print("   computed numerically (2-D Gauss-Legendre on the")
    print("   task-work MGF); the prefactor C_exp is UNSPECIFIED -")
    print("   never report it as a probability bound.")


if __name__ == "__main__":
    main()
