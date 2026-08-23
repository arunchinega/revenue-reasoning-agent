"""
capabilities/leakage.py — Revenue Leakage Detection.

Two layers, combined:
  RULE layer     — domain-activated business rules (utilities/insurance/banking),
                   each hit carries an estimated $ impact
  STATISTICAL    — anomaly-consensus votes scoped to revenue/rate features
Confidence tiers combine both: rule hit + statistical votes rank highest.
Output: leakage candidates ranked by estimated impact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.state import RunState


# ----------------------------------------------------------------------------
# rule implementations — each returns (mask, impact_series, description)
# ----------------------------------------------------------------------------

def _rule_billed_vs_expected_delta(df, cm, roles):
    if "billed_vs_expected_delta" not in df.columns:
        return None
    delta = df["billed_vs_expected_delta"]
    # under-billing: billed meaningfully below expected (tolerance 2% of expected)
    exp = df.get("expected_revenue")
    tol = (exp.abs() * 0.02).clip(lower=1.0) if exp is not None else 1.0
    mask = delta < -tol
    impact = (-delta).where(mask, 0.0)
    return mask, impact, "billed amount below price x quantity (under-billing)"


def _rule_zero_billed_with_usage(df, cm, roles):
    target, qty = cm["target_column"], roles.get("qty_col")
    if not qty or qty not in df.columns:
        return None
    mask = (df[target].fillna(0) == 0) & (df[qty].fillna(0) > 0)
    exp = df.get("expected_revenue")
    impact = (exp.where(mask, 0.0) if exp is not None
              else df[qty].where(mask, 0.0))
    return mask, impact, "zero billed amount despite recorded usage/quantity"


def _rule_tariff_misapplication(df, cm, roles):
    price = roles.get("price_col")
    if not price or price not in df.columns:
        return None
    seg_col = next((c for c in df.columns
                    if (pd.api.types.is_object_dtype(df[c])
                        or pd.api.types.is_string_dtype(df[c]))
                    and 2 <= df[c].nunique() <= 20), None)
    if seg_col is None:
        return None
    # modal rate per segment = contracted rate; deviations below it leak revenue
    modal = df.groupby(seg_col)[price].transform(lambda s: s.mode().iloc[0])
    mask = (df[price] < modal * 0.99)
    qty = roles.get("qty_col")
    impact = ((modal - df[price]) * df[qty]).where(mask, 0.0) if qty and qty in df.columns \
        else (modal - df[price]).where(mask, 0.0)
    return mask, impact.clip(lower=0), f"rate below the modal rate for its '{seg_col}' group"


def _rule_duplicate_billing(df, cm, roles):
    id_col, date_col, target = cm.get("id_column"), cm.get("date_column"), cm["target_column"]
    if not id_col or not date_col:
        return None
    dup = df.duplicated(subset=[id_col, date_col, target], keep="first")
    impact = df[target].where(dup, 0.0).abs()
    return dup, impact, "duplicate (entity, date, amount) billing row"


def _rule_excessive_discount(df, cm, roles):
    """Discounts far beyond the business's own norm — abuse, not promotion.
    Threshold: p99 of observed discounts, floored at 45%. Impact = revenue
    given away beyond the median discount level."""
    import pandas as pd
    target = cm["target_column"]
    disc_col = next((c for c in df.columns
                     if "discount" in c.lower()
                     and pd.api.types.is_numeric_dtype(df[c])), None)
    if disc_col is None:
        return None
    d = df[disc_col].fillna(0)
    if (d > 0).sum() < 10:
        return None
    thresh = max(45.0, float(d[d > 0].quantile(0.99)))
    mask = d > thresh
    if not mask.any():
        return mask, d.where(mask, 0.0) * 0.0, f"no discount above {thresh:.0f}%"
    med = float(d[(d > 0) & ~mask].median() or 0.0)
    base = df[target] / (1 - d / 100).clip(lower=0.05)
    impact = (base * (d - med) / 100).where(mask, 0.0).clip(lower=0)
    return mask, impact, (f"discount above {thresh:.0f}% (norm median {med:.0f}%) — "
                          f"revenue given away beyond policy")


_RULES = {
    "billed_vs_expected_delta": _rule_billed_vs_expected_delta,
    "zero_billed_with_usage": _rule_zero_billed_with_usage,
    "tariff_misapplication": _rule_tariff_misapplication,
    "unbilled_usage": _rule_zero_billed_with_usage,      # alias — same fn, deduped
    "duplicate_billing": _rule_duplicate_billing,
    "excessive_discount": _rule_excessive_discount,
    # insurance/banking rules reuse the generic mechanics for POC:
    "premium_vs_policy_delta": _rule_billed_vs_expected_delta,
    "duplicate_claims": _rule_duplicate_billing,
    "fee_schedule_mismatch": _rule_tariff_misapplication,
    "excess_waivers": _rule_zero_billed_with_usage,
}


def run_leakage_capability(state: RunState) -> dict:
    df = state.feature_df if state.feature_df is not None else state.raw_df
    cm = state.column_map
    roles = (state.feature_report or {}).get("detected_roles", {})
    active_rules = (state.domain.get("profile") or {}).get(
        "leakage_rules", ["billed_vs_expected_delta", "zero_billed_with_usage",
                          "excessive_discount", "duplicate_billing"])

    # statistical layer: reuse anomaly votes if present
    anomaly_votes: dict[int, int] = {}
    for rec in (state.results.get("anomaly") or {}).get("flagged", []):
        anomaly_votes[rec["row_index"]] = rec["votes"]

    rule_hits: dict[int, list[str]] = {}
    rule_impact: dict[int, float] = {}
    rules_fired: dict[str, dict] = {}
    seen = set()
    for rule_name in active_rules:
        fn = _RULES.get(rule_name)
        if fn is None or id(fn) in seen:      # skip unknown + alias double-fires
            continue
        seen.add(id(fn))
        out = fn(df, cm, roles)
        if out is None:
            rules_fired[rule_name] = {"applicable": False}
            continue
        mask, impact, desc = out
        n = int(mask.sum())
        rules_fired[rule_name] = {
            "applicable": True, "hits": n, "description": desc,
            "impact_total": round(float(impact.sum()), 2),
        }
        for idx in df.index[mask]:
            rule_hits.setdefault(int(idx), []).append(rule_name)
            # max, not sum: rules are alternative explanations of the SAME
            # missing money — a zero-bill row tripping both 'unbilled' and
            # 'under-billing' leaked its expected revenue once, not twice
            rule_impact[int(idx)] = max(rule_impact.get(int(idx), 0.0),
                                        float(impact.loc[idx]))

    # combine layers into ranked candidates
    candidates = []
    for idx, rules in rule_hits.items():
        votes = anomaly_votes.get(idx, 0)
        confidence = ("high" if votes >= 2 or len(rules) >= 2
                      else "medium" if votes == 1 or rule_impact.get(idx, 0) > 0
                      else "review")
        rec = {
            "row_index": idx,
            "rules": rules,
            "statistical_votes": votes,
            "confidence": confidence,
            "impact_estimate": round(rule_impact.get(idx, 0.0), 2),
        }
        if cm.get("id_column") and cm["id_column"] in df.columns:
            rec["entity"] = str(df.loc[idx, cm["id_column"]])
        if cm.get("date_column"):
            rec["date"] = str(df.loc[idx, cm["date_column"]])
        candidates.append(rec)
    candidates.sort(key=lambda r: r["impact_estimate"], reverse=True)

    total_impact = round(sum(r["impact_estimate"] for r in candidates), 2)
    out = {
        "domain": state.domain.get("name", "generic"),
        "rules_fired": rules_fired,
        "candidates": candidates[:200],
        "candidate_count": len(candidates),
        "total_impact_estimate": total_impact,
    }
    state.results["leakage"] = out
    state.ledger.log(
        stage="leakage", agent="deterministic",
        decision=(f"{len(candidates)} leakage candidate(s), "
                  f"estimated impact {total_impact:,.0f}"),
        reasoning=("rule layer x statistical consensus; rules active for domain "
                   f"'{out['domain']}': "
                   + ", ".join(f"{k}({v.get('hits', 0)})"
                               for k, v in rules_fired.items() if v.get("applicable"))),
        evidence=["domain.leakage_rules", "anomaly.votes",
                  "feature.billed_vs_expected_delta"],
        data={"total_impact_estimate": total_impact,
              "by_rule": {k: v.get("impact_total") for k, v in rules_fired.items()
                          if v.get("applicable")}},
    )
    return out
