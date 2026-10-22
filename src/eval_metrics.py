"""Shared contract-compliance metrics for the contract-debt project."""

CONTRACTS = {
    "K0_base": {"eta_min": [0.38, 0.55], "t_max": [2.00, 1.75]},
    "K1_loose_eta": {"eta_min": [0.35, 0.52], "t_max": [2.00, 1.75]},
    "K2_tight_eta": {"eta_min": [0.41, 0.58], "t_max": [2.00, 1.75]},
    "K3_tight_dly": {"eta_min": [0.38, 0.55], "t_max": [1.85, 1.55]},
}


def contract_metrics(tier_completion, tier_norm_delay, eta_min, t_max,
                     contract_stationary=None, protected_low=2):
    """Recompute completion, delay and FullStrict status from raw metrics.

    ``valid_full_contract_met`` is deliberately false when stationarity is
    unknown. Callers can inspect ``stationarity_known`` rather than silently
    treating missing evidence as a pass.
    """
    if tier_completion is None or tier_norm_delay is None:
        raise ValueError("tier completion and normalized delay are required")
    n = min(protected_low, len(eta_min), len(t_max),
            len(tier_completion), len(tier_norm_delay))
    if n <= 0:
        raise ValueError("at least one protected tier is required")

    eta_margin = [float(tier_completion[p]) - float(eta_min[p])
                  for p in range(n)]
    delay_margin = [float(t_max[p]) - float(tier_norm_delay[p])
                    for p in range(n)]
    completion_met = all(m >= 0.0 for m in eta_margin)
    delay_met = all(m >= 0.0 for m in delay_margin)
    full = completion_met and delay_met
    stationarity_known = contract_stationary is not None
    valid = full and bool(contract_stationary) if stationarity_known else False
    return {
        "completion_met": completion_met,
        "delay_met": delay_met,
        "full_contract_met": full,
        "valid_full_contract_met": valid,
        "eta_margin": eta_margin,
        "delay_margin": delay_margin,
        "stationarity_known": stationarity_known,
    }


def metrics_for_record(record, contract):
    """Evaluate one serialized result record under ``contract``."""
    spec = CONTRACTS[contract] if isinstance(contract, str) else contract
    return contract_metrics(
        record.get("tier_completion"),
        record.get("tier_norm_delay"),
        spec["eta_min"],
        spec["t_max"],
        record.get("contract_stationary"),
    )
