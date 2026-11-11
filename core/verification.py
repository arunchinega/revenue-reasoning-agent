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

    # CoVe v2: financial and claim-level business invariants.
    if lk and not lk.get("error"):
        rules = lk.get("rules_fired") or {}
        applicable = [v for v in rules.values() if isinstance(v, dict)
                      and v.get("applicable") and v.get("impact_total") is not None]
        if applicable and _finite_number(lk.get("total_impact_estimate")):
            values = [v.get("impact_total") for v in applicable]
            if not all(_finite_number(v) and float(v) >= 0 for v in values):
                check("leakage_rule_impacts", "rejected",
                      "Rule impact totals must be finite and nonnegative",
                      "results.leakage.rules_fired")
            else:
                gross = sum(float(v) for v in values)
                net = float(lk["total_impact_estimate"])
                if gross + 0.01 < net:
                    check("leakage_reconciliation", "rejected",
                          "Deduplicated total exceeds sum of nonnegative rule totals",
                          "results.leakage.rules_fired")
                elif abs(gross - net) > 0.01:
                    check("leakage_reconciliation", "unverified",
                          "Gross rule impacts differ from deduplicated candidate impact; "
                          "overlap must be quantified before claiming a reconciled breakdown",
                          "results.leakage.rules_fired")
                else:
                    check("leakage_reconciliation", "verified",
                          "Rule totals reconcile with candidate total within rounding tolerance",
                          "results.leakage.rules_fired")
        candidates = lk.get("candidates")
        count = lk.get("candidate_count")
        if count is not None and (not isinstance(count, int) or count < 0):
            check("leakage_candidate_count", "rejected",
                  "Candidate count must be a nonnegative integer",
                  "results.leakage.candidate_count")
        elif isinstance(candidates, list) and isinstance(count, int):
            # The capability intentionally truncates detailed candidates to top 200.
            check("leakage_candidate_count", "verified" if len(candidates) <= count else "rejected",
                  "Displayed candidate sample must not exceed reported total",
                  "results.leakage.candidates")
        check("leakage_claim_language", "unverified",
              "Estimated exposure must be labelled candidate/potential, not recoverable, "
              "until contracts, payment records and adjudication support recovery",
              "results.leakage")

    if fc and not fc.get("error"):
        verdict = str(fc.get("verdict", ""))
        if verdict in ("escalate", "rejected", "reject"):
            check("forecast_claim_language", "unverified",
                  "Escalated/rejected forecast is illustrative only, never certified",
                  "results.forecasting.verdict")
        # An explicit feature manifest is needed to audit prediction-time availability.
        manifest = fc.get("feature_availability") or fc.get("feature_contract")
        if not manifest:
            check("forecast_feature_availability", "unverified",
                  "No per-feature prediction-time availability manifest attached to forecast",
                  "results.forecasting")
        elif isinstance(manifest, dict):
            unknown = [k for k, v in manifest.items()
                       if v not in (True, "known_future", "historical_lag", "recursive")]
            check("forecast_feature_availability",
                  "unverified" if unknown else "verified",
                  "Unproven future availability: " + ", ".join(map(str, unknown[:8]))
                  if unknown else "All declared features have an approved availability class",
                  "results.forecasting.feature_availability")
        else:
            check("forecast_feature_availability", "unverified",
                  "Feature availability manifest must be a mapping",
                  "results.forecasting")

    rca = state.results.get("rca", {})
    if rca and not rca.get("error"):
        check("rca_causality", "unverified",
              "Ranked decomposition hypotheses are not established causal explanations",
              "results.rca.hypotheses")

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
