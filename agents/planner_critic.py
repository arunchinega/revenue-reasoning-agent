"""
agents/planner_critic.py — the reasoning loop for the forecasting capability.

Planner: selects candidates from the registry citing eligibility evidence.
Critic:  judges backtest metrics → accept_single | accept_ensemble | retry | escalate.
Both have deterministic fallbacks so the loop runs (and is testable) without Ollama.
Loop:    plan → execute → critique, max N retries, everything ledger-logged.
"""
from __future__ import annotations

import json
import numpy as np

from core.llm import MODELS, call_json
from core.state import RunState
from capabilities.forecasting import (
    CandidateResult, metrics_table, prepare_series, registry_eligibility,
    run_backtest,
)

MAX_RETRIES = 2
ACCEPT_MAPE = 20.0          # acceptance criterion (Planner may override)
ENSEMBLE_CORR_MAX = 0.9     # blend only when errors aren't near-duplicates


# ----------------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------------

def _fallback_plan(eligibility: dict) -> dict:
    """Diversity-aware deterministic selection: baseline floor + the first
    eligible candidate from each family, so the bake-off spans statistical /
    decomposition / ML / deep-learning even without the LLM Planner."""
    picks, families_taken = ["seasonal_naive"], set()
    reasons = []
    for name, v in eligibility.items():
        fam = v["family"]
        if name == "seasonal_naive" or not v["eligible"] or fam in families_taken:
            continue
        picks.append(name)
        families_taken.add(fam)
        reasons.append(f"{name} [{fam}]: {v['reasoning']}")
        if len(picks) >= 5:      # floor + up to 4 diverse challengers
            break
    return {
        "candidates": picks,
        "preprocessing": {"fill_gaps": True},
        "acceptance_mape": ACCEPT_MAPE,
        "reasoning": ("deterministic fallback, family-diverse selection: "
                      + "; ".join(reasons)),
    }


def _sanitize_llm_plan(plan: dict) -> dict:
    """LLMs return creative shapes — coerce every field to its contract.
    (CoT prompts especially invite reasoning-as-list-of-steps.)"""
    import re as _re
    if not isinstance(plan, dict):
        return {}
    r = plan.get("reasoning", "")
    if isinstance(r, (list, tuple)):
        r = " → ".join(str(x) for x in r)
    plan["reasoning"] = str(r)
    cands = plan.get("candidates", [])
    flat = []
    for c in cands if isinstance(cands, (list, tuple)) else []:
        if isinstance(c, dict):
            c = c.get("model") or c.get("name") or next(iter(c.values()), None)
        if isinstance(c, str) and c.strip():
            flat.append(c.strip())
    plan["candidates"] = flat
    am = plan.get("acceptance_mape")
    if not isinstance(am, (int, float)):
        m = _re.search(r"[\d.]+", str(am or ""))
        plan["acceptance_mape"] = float(m.group()) if m else None
    pp = plan.get("preprocessing")
    if isinstance(pp, (list, tuple)):
        pp = {str(k): True for k in pp}
    plan["preprocessing"] = pp if isinstance(pp, dict) else {}
    return plan


def plan_forecasting(state: RunState, use_llm: bool = True) -> dict:
    ts_ev = (state.eda_report.get("timeseries") or {})
    eligibility = registry_eligibility(ts_ev)

    plan = None
    if use_llm:
        llm = call_json(
            role="reasoner",
            system=(
                "You are the Planner of a forecasting agent. FIRST think step by step ""in the 'reasoning' field — walk the EDA evidence, then eliminate, ""then select. Select 3-4 candidate "
                "models from the eligible registry entries (ALWAYS include "
                "seasonal_naive as the floor). Cite EDA evidence for each pick. "
                'Respond ONLY with JSON: {"candidates": [str], '
                '"preprocessing": {"winsorize": bool, "fill_gaps": bool}, '
                '"acceptance_mape": float, '
                '"reasoning": str (cite evidence keys for each candidate)}'
            ),
            user=(
                f"EDA TIMESERIES EVIDENCE:\n{json.dumps(ts_ev, default=str)}\n\n"
                f"REGISTRY ELIGIBILITY:\n{json.dumps(eligibility)}\n\n"
                f"TOP FEATURES:\n{state.feature_report.get('top_features', [])[:8]}"
            ),
            required_keys=("candidates",),
            fallback=lambda: _fallback_plan(eligibility),
        )
        plan = _sanitize_llm_plan(llm.parsed or {})
        if plan.get("acceptance_mape") is None:
            plan["acceptance_mape"] = ACCEPT_MAPE
        # sanitize: only eligible registry names, floor always present
        plan["candidates"] = [c for c in plan.get("candidates", [])
                              if eligibility.get(c, {}).get("eligible")]
        if "seasonal_naive" not in plan["candidates"]:
            plan["candidates"].insert(0, "seasonal_naive")
        # POOL FLOOR: the LLM may sharpen the pool but never weaken it below
        # the family-diverse fallback — union the two so at least one model
        # per eligible family always races
        floor = _fallback_plan(eligibility)["candidates"]
        added = [c for c in floor if c not in plan["candidates"]]
        if added:
            plan["candidates"] += added
            plan["reasoning"] = (plan.get("reasoning", "") +
                                 f" | pool floor added family coverage: {added}")
        if len(plan["candidates"]) < 2:
            plan = _fallback_plan(eligibility)
        plan.setdefault("preprocessing", {"fill_gaps": True})
        plan.setdefault("acceptance_mape", ACCEPT_MAPE)
        agent = "deterministic" if llm.used_fallback else MODELS["reasoner"]
    else:
        plan = _fallback_plan(eligibility)
        agent = "deterministic"

    excluded = {}
    for n, v in eligibility.items():
        if n in plan["candidates"]:
            continue
        excluded[n] = (v["reasoning"] if not v["eligible"]
                       else f"eligible ({v['reasoning']}) but not selected this round")

    # full-roster status snapshot (standardized boards consume this)
    plan["roster"] = {
        n: {"family": v["family"],
            "status": ("selected" if n in plan["candidates"]
                       else "data_gated" if not v["eligible"]
                       else "not_selected"),
            "reason": v["reasoning"]}
        for n, v in eligibility.items()
    }
    state.ledger.log(
        stage="planner", agent=agent,
        decision=f"Forecast candidates: {plan['candidates']} "
                 f"(accept if MAPE < {plan['acceptance_mape']})",
        reasoning=plan.get("reasoning", ""),
        evidence=["eda.timeseries", "feature.top_features"],
        data={"excluded": excluded, "preprocessing": plan["preprocessing"]},
    )
    state.plan["forecasting"] = plan
    return plan


# ----------------------------------------------------------------------------
# Critic
# ----------------------------------------------------------------------------

def _ensemble_check(results: dict[str, CandidateResult]) -> tuple[dict | None, str]:
    """Weighted inverse-MAPE blend of top-2 when errors aren't near-duplicates.
    Returns (ensemble_or_None, evaluation_reason) — the reason is ALWAYS
    produced so boards can show why an ensemble did or didn't happen."""
    ok = [r for r in results.values()
          if r.forecast is not None and np.isfinite(r.mape) and r.family != "baseline"]
    ok.sort(key=lambda r: r.mape)
    if len(ok) < 2:
        return None, "fewer than two non-baseline candidates produced valid forecasts"
    a, b = ok[0], ok[1]
    if a.backtest_preds is None or b.backtest_preds is None:
        return None, "top candidates lack aligned backtest predictions to correlate"
    n = min(len(a.backtest_preds), len(b.backtest_preds))
    err_a = a.backtest_preds[:n] - a.backtest_actuals[:n]
    err_b = b.backtest_preds[:n] - b.backtest_actuals[:n]
    if err_a.std() < 1e-9 or err_b.std() < 1e-9:
        return None, "a top candidate has near-zero error variance — blending is meaningless"
    corr = float(np.corrcoef(err_a, err_b)[0, 1])
    if corr >= ENSEMBLE_CORR_MAX:
        return None, (f"top-2 ({a.name}, {b.name}) error correlation "
                      f"{corr:.3f} ≥ {ENSEMBLE_CORR_MAX} — members too similar to gain")
    wa = (1 / a.mape) / (1 / a.mape + 1 / b.mape)
    blend_bt = wa * a.backtest_preds[:n] + (1 - wa) * b.backtest_preds[:n]
    from capabilities.forecasting import mape as _mape
    blend_mape = _mape(a.backtest_actuals[:n], blend_bt)
    if blend_mape >= a.mape:
        return None, (f"blend MAPE {blend_mape:.2f}% did not beat best single "
                      f"{a.name} ({a.mape:.2f}%) despite corr {corr:.3f}")
    m = min(len(a.forecast), len(b.forecast))
    return {
        "members": [a.name, b.name], "weights": [round(wa, 3), round(1 - wa, 3)],
        "error_corr": round(corr, 3), "blend_mape": round(blend_mape, 2),
        "best_single_mape": round(a.mape, 2),
        "forecast": (wa * a.forecast[:m] + (1 - wa) * b.forecast[:m]),
    }, (f"blend of {a.name}+{b.name} wins: MAPE {blend_mape:.2f}% < "
        f"{a.mape:.2f}% (corr {corr:.3f})")


def critique_forecasting(state: RunState, results: dict[str, CandidateResult],
                         plan: dict, attempt: int, use_llm: bool = True) -> dict:
    table = metrics_table(results)
    accept_mape = float(plan.get("acceptance_mape", ACCEPT_MAPE))
    valid = [r for r in table if r["mape"] is not None]
    best = valid[0] if valid else None
    baseline = next((r for r in valid if r["family"] == "baseline"), None)
    ensemble, ensemble_reason = _ensemble_check(results)
    # (attached to the verdict below so the result assembly can persist it)

    # --- deterministic verdict (always computed; LLM only narrates on top) ---
    if best is None:
        verdict = {"verdict": "escalate",
                   "reasoning": "no candidate produced valid backtest metrics"}
    elif ensemble and ensemble["blend_mape"] < min(best["mape"], accept_mape):
        verdict = {
            "verdict": "accept_ensemble", "ensemble": ensemble,
            "reasoning": (
                f"blend of {ensemble['members']} (weights {ensemble['weights']}) "
                f"reaches MAPE {ensemble['blend_mape']}% vs best single "
                f"{ensemble['best_single_mape']}%; error corr {ensemble['error_corr']} "
                f"< {ENSEMBLE_CORR_MAX} so members are complementary"
            ),
        }
    elif best["mape"] <= accept_mape and (baseline is None or best["mape"] < baseline["mape"]):
        verdict = {
            "verdict": "accept_single", "winner": best["model"],
            "reasoning": (
                f"{best['model']} MAPE {best['mape']}% meets criterion "
                f"(< {accept_mape}%) and beats baseline "
                f"({baseline['mape']}%)" if baseline else ""
            ),
        }
    elif attempt < MAX_RETRIES:
        # name the actual failure: threshold miss vs failure to beat the floor
        if best["mape"] > accept_mape:
            failure = f"best MAPE {best['mape']}% exceeds criterion {accept_mape}%"
        else:
            failure = (f"best MAPE {best['mape']}% is within criterion but does not "
                       f"beat the seasonal-naive floor ({baseline['mape']}%)"
                       if baseline else
                       f"best MAPE {best['mape']}% lacks a baseline comparison")
        # diagnose: outliers in EDA → winsorize retry; else widen candidate pool
        outlier_heavy = any(
            v.get("outlier_pct_iqr", 0) > 3
            for v in (state.eda_report.get("numeric") or {}).values()
        )
        change = ({"preprocessing": {"winsorize": True, "fill_gaps": True}}
                  if outlier_heavy and not plan["preprocessing"].get("winsorize")
                  else {"widen_pool": True})
        verdict = {
            "verdict": "retry", "change": change,
            "reasoning": (
                failure + "; "
                + ("EDA shows outlier-heavy target → retry with winsorization"
                   if "preprocessing" in change
                   else "no preprocessing lever left → widen candidate pool")
            ),
        }
    else:
        verdict = {
            "verdict": "escalate", "best_available": best["model"],
            "reasoning": f"retries exhausted; best available {best['model']} "
                         f"at MAPE {best['mape']}% — human decision needed",
        }

    verdict["ensemble_eval_reason"] = ensemble_reason

    # optional LLM narrative enrichment (never changes the verdict)
    if use_llm:
        llm = call_json(
            role="reasoner",
            system=(
                "You are the Critic of a forecasting agent. A deterministic rule "
                "engine produced a verdict; write a sharper one-paragraph reasoning "
                "referencing the metrics. Do NOT change the verdict. Respond ONLY "
                'with JSON: {"reasoning": str}'
            ),
            user=f"METRICS:\n{json.dumps(table)}\n\nVERDICT:\n{json.dumps({k: v for k, v in verdict.items() if k != 'ensemble'}, default=str)}",
            required_keys=("reasoning",),
            fallback=lambda: {"reasoning": verdict["reasoning"]},
        )
        if llm.parsed and not llm.used_fallback and "reasoning" in llm.parsed:
            r = llm.parsed["reasoning"]
            if isinstance(r, (list, tuple)):
                r = " → ".join(str(x) for x in r)
            if isinstance(r, str) and r.strip():
                verdict["reasoning"] = r

    state.ledger.log(
        stage="critic", agent=MODELS["reasoner"] if use_llm else "deterministic",
        decision=f"[attempt {attempt + 1}] {verdict['verdict']}"
                 + (f" → {verdict.get('winner')}" if verdict.get("winner") else "")
                 + (f" → blend {verdict['ensemble']['members']}" if verdict.get("ensemble") else ""),
        reasoning=verdict["reasoning"],
        evidence=["backtest.metrics"],
        hitl_required=verdict["verdict"] == "escalate",
        data={"metrics": table},
    )
    return verdict


# ----------------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------------

def run_forecasting_capability(state: RunState, use_llm: bool = True,
                               horizon: int = 30) -> dict:
    cm = state.column_map
    ts_ev = state.eda_report.get("timeseries") or {}
    period = (ts_ev.get("seasonality") or {}).get("period", 7)

    plan = plan_forecasting(state, use_llm=use_llm)

    for attempt in range(MAX_RETRIES + 1):
        ts = prepare_series(state.raw_df, cm["date_column"], cm["target_column"],
                            plan.get("preprocessing"))
        results = run_backtest(ts, plan["candidates"], period=period, horizon=horizon)
        verdict = critique_forecasting(state, results, plan, attempt, use_llm=use_llm)

        if verdict["verdict"] in ("accept_single", "accept_ensemble", "escalate"):
            winner = verdict.get("winner")
            ens = verdict.get("ensemble") or {}
            best_avail = verdict.get("best_available")
            forecast = (ens["forecast"] if ens else
                        results[winner].forecast if winner and results.get(winner)
                        else results[best_avail].forecast
                        if best_avail and results.get(best_avail) else None)
            table = metrics_table(results)
            # merge execution outcomes into the planner's roster snapshot
            roster = dict(plan.get("roster", {}))
            for m in table:
                if m["model"] in roster:
                    roster[m["model"]] = {
                        **roster[m["model"]],
                        "status": ("failed" if m.get("error") or m.get("mape") is None
                                   else "ran"),
                        "reason": (m.get("error") or roster[m["model"]]["reason"]
                                   if (m.get("error") or m.get("mape") is None)
                                   else roster[m["model"]]["reason"]),
                        "mape": m.get("mape"), "rmse": m.get("rmse"),
                        "folds": m.get("folds"),
                    }
            # winner audit card — same shape for single / ensemble / escalate
            w_name = winner or (ens.get("members") if ens else best_avail)
            best_row = next((m for m in table if m["model"] == (winner or best_avail)
                             and m.get("mape") is not None), None)
            base_row = next((m for m in table if m["family"] == "baseline"
                             and m.get("mape") is not None), None)
            runner = next((m for m in table
                           if m["model"] not in ([winner] if winner else [])
                           and m["family"] != "baseline"
                           and m.get("mape") is not None), None)
            card_mape = (ens["blend_mape"] if ens else
                         best_row["mape"] if best_row else None)
            caveats = []
            if verdict["verdict"] == "escalate":
                caveats.append("NOT certified — shown as best attempt only")
            if card_mape is not None and base_row and \
                    (base_row["mape"] - card_mape) < 0.5:
                caveats.append(f"wins by only "
                               f"{base_row['mape'] - card_mape:.2f}pp over the "
                               f"naive floor — weak certification")
            out = {
                "verdict": verdict["verdict"],
                "winner": winner or (ens.get("members") if ens else None),
                "winner_label": (f"ensemble({'+'.join(ens['members'])})" if ens
                                 else winner or
                                 (f"escalated (best attempt: {best_avail})"
                                  if best_avail else "escalated")),
                "ensemble": ({k: ens[k] for k in
                              ("members", "weights", "error_corr", "blend_mape")}
                             if ens else None),
                "ensemble_eval": {"fired": bool(ens),
                                  "reason": verdict.get("ensemble_eval_reason", "")},
                "roster": roster,
                "winner_card": {
                    "name": w_name, "verdict": verdict["verdict"],
                    "why_selected": verdict.get("reasoning", ""),
                    "mape": card_mape,
                    "rmse": (best_row or {}).get("rmse"),
                    "folds": (best_row or {}).get("folds"),
                    "margin_vs_baseline": (round(base_row["mape"] - card_mape, 2)
                                           if card_mape is not None and base_row
                                           else None),
                    "margin_vs_runner_up": (round(runner["mape"] - card_mape, 2)
                                            if card_mape is not None and runner
                                            else None),
                    "preprocessing": plan.get("preprocessing", {}),
                    "attempts": attempt + 1,
                    "caveats": caveats,
                },
                "metrics": table,
                "forecast": forecast.tolist() if forecast is not None else None,
                "best_available": best_avail,
                "critic_reasoning": verdict.get("reasoning", ""),
                "attempts": attempt + 1,
            }
            state.results["forecasting"] = out
            return out

        # retry: apply the Critic's change
        change = verdict.get("change", {})
        if "preprocessing" in change:
            plan["preprocessing"] = change["preprocessing"]
        if change.get("widen_pool"):
            elig = registry_eligibility(ts_ev)
            extra = [n for n, v in elig.items()
                     if v["eligible"] and n not in plan["candidates"]]
            plan["candidates"] += extra[:2]

    # should not reach here, but stay safe
    out = {"verdict": "escalate", "metrics": metrics_table(results), "attempts": MAX_RETRIES + 1}
    state.results["forecasting"] = out
    return out
