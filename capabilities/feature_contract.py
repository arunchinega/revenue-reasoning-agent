"""Forecast-time feature eligibility contract.

A column's presence in historical data does not make it safe to use at
prediction time. Explicitly classify availability; unknown means excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

KNOWN_FUTURE = "known_future"
HISTORICAL_ONLY = "historical_only"
UNKNOWN = "unknown"

CALENDAR_FEATURES = frozenset({"dow", "month", "is_month_end", "day_of_month",
                                "day_of_year", "week_of_year", "is_weekend"})
HISTORICAL_PREFIXES = ("customer_revenue_lag", "customer_revenue_roll",
                       "revenue_lag", "revenue_roll")


@dataclass(frozen=True)
class FeatureEligibility:
    name: str
    availability: str
    reason: str


def classify_feature(name: str, *, target_column: str,
                     future_known: Iterable[str] = ()) -> FeatureEligibility:
    """Fail closed: explicitly declared future-known features must be reviewed.

    A caller-provided future_known list is an assertion, not evidence of
    actual future availability; training/serving checks remain mandatory.
    """
    if name == target_column:
        return FeatureEligibility(name, HISTORICAL_ONLY, "Target value cannot be a predictor at forecast time")
    if name in CALENDAR_FEATURES:
        return FeatureEligibility(name, KNOWN_FUTURE, "Derivable from the forecast date")
    if name.startswith(HISTORICAL_PREFIXES) or name.startswith(target_column + "_lag") or name.startswith(target_column + "_roll"):
        return FeatureEligibility(name, HISTORICAL_ONLY, "Requires historical target values and recursive or lag-aware prediction")
    if name in {"expected_revenue", "billed_vs_expected_delta", "revenue_per_unit",
                "rfm_monetary", "rfm_frequency", "rfm_recency_days"}:
        return FeatureEligibility(name, HISTORICAL_ONLY, "Derived from realized transactions or full-history aggregates")
    if name in future_known:
        return FeatureEligibility(name, KNOWN_FUTURE, "Caller asserts known future values; validate future coverage before use")
    return FeatureEligibility(name, UNKNOWN, "Future availability not established; exclude by default")


def select_known_future_features(columns: Iterable[str], *,
                                 target_column: str,
                                 future_known: Iterable[str] = ()) -> list[str]:
    return [name for name in columns if classify_feature(
        name, target_column=target_column, future_known=future_known
    ).availability == KNOWN_FUTURE]
