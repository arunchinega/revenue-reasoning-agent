"""Deterministic Chain-of-Verification gate for analytical results.

This is a first-stage verifier, not proof of causal validity or commercial readiness.
No LLM is trusted to approve its own quantitative claims.
"""
from __future__ import annotations

import math
from typing import Any


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def verify_results(state) -> dict:
    """Verify basic numeric invariants; do not silently certify unsupported claims."""
    checks: list[dict] = []

    def check(claim: str, status: str, reason: str, evidence: str) -> None:
        checks.append(dict(claim=claim, status=status, reason=reason, evidence=evidence))

    fc = state.results.get("forecasting", {})
    if fc and not fc.get("error"):
        forecast = fc.get("forecast")
        if forecast is None:
            check("forecast_output", "unverified", "No forecast values returned", "results.forecasting.forecast")
        elif not isinstance(forecast, (list, tuple)) or not forecast or not all(_finite_number(x) for x in forecast):
            check("forecast_output", "rejected", "Forecast must be a nonempty sequence of finite values", "results.forecasting.forecast")
        else:
            check("forecast_output", "verified", "Forecast contains finite numeric values", "results.forecasting.forecast")
        metrics = fc.get("metrics", [])
        for m in metrics:
            if m.get("error"):
                continue
            value = m.get("mape")
            if value is not None:
                check("mape:" + str(m.get("model", "?")),
                      "verified" if _finite_number(value) and float(value) >= 0 else "rejected",
                      "MAPE is finite and nonnegative" if _finite_number(value) and float(value) >= 0 else "Invalid MAPE",
                      "results.forecasting.metrics")
        if fc.get("verdict") in ("accept_single", "accept_ensemble"):
            check("forecast_certification", "unverified",
                  "Independent chronological holdout and uncertainty calibration are not checked by this gate",
                  "results.forecasting")

    lk = state.results.get("leakage", {})
    if lk and not lk.get("error"):
        impact = lk.get("total_impact_estimate")
        if impact is not None:
            check("leakage_impact", "verified" if _finite_number(impact) and float(impact) >= 0 else "rejected",
                  "Nonnegative finite estimate" if _finite_number(impact) and float(impact) >= 0 else "Invalid leakage estimate",
                  "results.leakage.total_impact_estimate")
        check("leakage_recoverability", "unverified",
              "A candidate leakage estimate is not proof that money is recoverable",
              "results.leakage")

    status = ("rejected" if any(c["status"] == "rejected" for c in checks)
              else "qualified" if any(c["status"] == "unverified" for c in checks)
              else "verified" if checks else "not_applicable")
    report = {"status": status, "checks": checks,
              "verified_count": sum(c["status"] == "verified" for c in checks),
              "rejected_count": sum(c["status"] == "rejected" for c in checks),
              "unverified_count": sum(c["status"] == "unverified" for c in checks)}
    state.results["verification"] = report
    if state.ledger is not None:
        state.ledger.log(stage="chain_of_verification", agent="deterministic",
                         decision="Verification status: " + status,
                         reasoning=f"{report['verified_count']} verified; {report['rejected_count']} rejected; "
                                   f"{report['unverified_count']} unverified",
                         evidence=[c["evidence"] for c in checks])
    return report
