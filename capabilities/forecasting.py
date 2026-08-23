"""
capabilities/forecasting.py — Forecasting model registry + backtest executor.

Registry pattern: every model declares a data-gate checked against EDA evidence.
The Planner selects candidates (or the deterministic gate-fallback does);
nothing is hardcoded to run. Unavailable libraries are excluded WITH REASONING
— exclusions are first-class ledger content.

All runners share one contract:
    fit_predict(train: pd.Series, horizon: int, exog_train, exog_future) -> np.ndarray
Backtest: rolling-origin with k folds → MAPE / RMSE per candidate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- optional heavy deps: guarded, absence is a logged exclusion -------------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


# ----------------------------------------------------------------------------
# runners — one function per registry model
# ----------------------------------------------------------------------------

def _run_seasonal_naive(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    if len(train) < period:
        return np.repeat(train.iloc[-1], horizon)
    last_cycle = train.iloc[-period:].to_numpy()
    reps = int(np.ceil(horizon / period))
    return np.tile(last_cycle, reps)[:horizon]


def _run_ets(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    seasonal = "add" if len(train) >= 2 * period + 1 else None
    model = ExponentialSmoothing(
        train.astype(float), trend="add",
        seasonal=seasonal, seasonal_periods=period if seasonal else None,
        initialization_method="estimated",
    ).fit()
    return model.forecast(horizon).to_numpy()


def _run_arima_auto(train: pd.Series, horizon: int, **_) -> np.ndarray:
    """Small AIC grid search — pmdarima-free so it runs anywhere."""
    best_aic, best_fit = np.inf, None
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                if p == q == 0:
                    continue
                try:
                    fit = ARIMA(train.astype(float), order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_fit = fit.aic, fit
                except Exception:  # noqa: BLE001
                    continue
    if best_fit is None:
        raise RuntimeError("no ARIMA order converged")
    return best_fit.forecast(horizon).to_numpy()


def _run_sarima(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    fit = SARIMAX(
        train.astype(float), order=(1, 1, 1),
        seasonal_order=(1, 1, 1, period),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return fit.forecast(horizon).to_numpy()


def _run_prophet(train: pd.Series, horizon: int, **_) -> np.ndarray:
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=len(train) > 400,
                daily_seasonality=False)
    m.fit(dfp)
    freq = pd.infer_freq(train.index) or "D"
    future = m.make_future_dataframe(periods=horizon, freq=freq)
    fc = m.predict(future)
    return fc["yhat"].to_numpy()[-horizon:]


def _make_lag_matrix(y: pd.Series, lags: tuple[int, ...]) -> pd.DataFrame:
    X = pd.DataFrame(index=y.index)
    for lag in lags:
        X[f"lag{lag}"] = y.shift(lag)
    X["dow"] = y.index.dayofweek if hasattr(y.index, "dayofweek") else 0
    return X


def _run_ml_lags(model, train: pd.Series, horizon: int, period: int = 7) -> np.ndarray:
    """Shared lag-feature fit + recursive multi-step forecast for sklearn-style
    regressors (XGBoost, GradientBoosting, RandomForest)."""
    lags = tuple(l for l in (1, 2, 3, period, 2 * period) if l < len(train) - 5)
    X = _make_lag_matrix(train, lags)
    mask = X.notna().all(axis=1)
    model.fit(X[mask], train[mask])
    history = train.copy()
    preds = []
    freq = pd.infer_freq(train.index) or "D"
    for _ in range(horizon):
        next_idx = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
        row = {f"lag{lag}": history.iloc[-lag] for lag in lags}
        row["dow"] = next_idx.dayofweek if hasattr(next_idx, "dayofweek") else 0
        p = float(model.predict(pd.DataFrame([row]))[0])
        preds.append(p)
        history.loc[next_idx] = p
    return np.array(preds)


def _run_xgboost_lags(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                             subsample=0.9, verbosity=0)
    return _run_ml_lags(model, train, horizon, period)


def _run_gradient_boosting_lags(train: pd.Series, horizon: int,
                                period: int = 7, **_) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.08, subsample=0.9,
                                      random_state=0)
    return _run_ml_lags(model, train, horizon, period)


def _run_random_forest_lags(train: pd.Series, horizon: int,
                            period: int = 7, **_) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=300, max_depth=8,
                                  random_state=0, n_jobs=-1)
    return _run_ml_lags(model, train, horizon, period)



def _run_sklearn_lags(model_factory, train: pd.Series, horizon: int,
                      period: int = 7) -> np.ndarray:
    """Shared recursive lag-feature forecaster for sklearn-style regressors."""
    lags = tuple(l for l in (1, 2, 3, period, 2 * period) if l < len(train) - 5)
    X = _make_lag_matrix(train, lags)
    mask = X.notna().all(axis=1)
    model = model_factory()
    model.fit(X[mask], train[mask])
    history = train.copy()
    preds = []
    freq = pd.infer_freq(train.index) or "D"
    for _ in range(horizon):
        next_idx = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
        row = {f"lag{lag}": history.iloc[-lag] for lag in lags}
        row["dow"] = next_idx.dayofweek if hasattr(next_idx, "dayofweek") else 0
        p = float(model.predict(pd.DataFrame([row]))[0])
        preds.append(p)
        history.loc[next_idx] = p
    return np.array(preds)


def _run_gradient_boosting(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    from sklearn.ensemble import GradientBoostingRegressor
    return _run_sklearn_lags(
        lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.08, subsample=0.9),
        train, horizon, period)


def _run_random_forest(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    return _run_sklearn_lags(
        lambda: RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1,
                                      random_state=0),
        train, horizon, period)


def _run_lstm(train: pd.Series, horizon: int, period: int = 7, **_) -> np.ndarray:
    from tensorflow import keras
    window = min(4 * period, max(len(train) // 4, period))
    vals = train.to_numpy(dtype=float)
    mean, std = vals.mean(), vals.std() or 1.0
    scaled = (vals - mean) / std
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    X = np.array(X)[..., None]
    y = np.array(y)
    model = keras.Sequential([
        keras.layers.Input((window, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=4,
                                                       restore_best_weights=True)])
    buf = list(scaled)
    preds = []
    for _ in range(horizon):
        window_in = np.array(buf[-window:])[None, :, None]
        p = float(model.predict(window_in, verbose=0)[0, 0])
        preds.append(p)
        buf.append(p)
    return np.array(preds) * std + mean


# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------

@dataclass
class RegistryModel:
    name: str
    family: str
    runner: Callable
    available: bool
    unavailable_reason: str = ""
    gate: Callable[[dict], tuple[bool, str]] = None  # (eligible, reasoning)


def _gate_always(ev: dict) -> tuple[bool, str]:
    return True, "always eligible"


def _gate_seasonality(ev: dict) -> tuple[bool, str]:
    seas = ev.get("seasonality") or {}
    if seas.get("detected"):
        return True, f"seasonality detected (period={seas['period']}, strength={seas['strength']})"
    return False, "no seasonality detected in EDA"


def _gate_prophet(ev: dict) -> tuple[bool, str]:
    seas = ev.get("seasonality") or {}
    if seas.get("detected") or ev.get("trend_direction") in ("increasing", "decreasing"):
        return True, f"seasonality={seas.get('detected', False)}, trend={ev.get('trend_direction')}"
    return False, "no seasonality or trend structure for decomposition to exploit"


def _gate_xgb(ev: dict) -> tuple[bool, str]:
    if ev.get("points", 0) >= 60:
        return True, f"{ev['points']} points sufficient for lag-feature learning"
    return False, f"only {ev.get('points', 0)} points — too few for lag matrix"


def _gate_lstm(ev: dict) -> tuple[bool, str]:
    pts = ev.get("points", 0)
    if pts > 400:
        return True, f"{pts} points > 400 — sufficient for a small LSTM"
    return False, f"{pts} points ≤ 400 — LSTM would overfit; excluded"


def build_registry() -> list[RegistryModel]:
    return [
        RegistryModel("seasonal_naive", "baseline", _run_seasonal_naive, True, gate=_gate_always),
        RegistryModel("ets", "statistical", _run_ets, HAS_STATSMODELS,
                      "statsmodels not installed", _gate_always),
        RegistryModel("arima_auto", "statistical", _run_arima_auto, HAS_STATSMODELS,
                      "statsmodels not installed", _gate_always),
        RegistryModel("sarima", "statistical", _run_sarima, HAS_STATSMODELS,
                      "statsmodels not installed", _gate_seasonality),
        RegistryModel("prophet", "decomposition", _run_prophet, HAS_PROPHET,
                      "prophet not installed", _gate_prophet),
        RegistryModel("xgboost_lags", "ml_boosting", _run_xgboost_lags, HAS_XGB,
                      "xgboost not installed", _gate_xgb),
        RegistryModel("gradient_boosting", "ml_boosting", _run_gradient_boosting,
                      True, gate=_gate_xgb),
        RegistryModel("random_forest", "ml_bagging", _run_random_forest,
                      True, gate=_gate_xgb),
        RegistryModel("lstm", "deep_learning", _run_lstm, HAS_TF,
                      "tensorflow not installed", _gate_lstm),
    ]


def registry_eligibility(ts_evidence: dict) -> dict[str, dict]:
    """For every registry model: eligible? why? — Planner input + ledger content."""
    out = {}
    for m in build_registry():
        if not m.available:
            out[m.name] = {"eligible": False, "family": m.family,
                           "reasoning": f"unavailable: {m.unavailable_reason}"}
            continue
        ok, why = m.gate(ts_evidence)
        out[m.name] = {"eligible": ok, "family": m.family, "reasoning": why}
    return out


# ----------------------------------------------------------------------------
# backtest executor
# ----------------------------------------------------------------------------

@dataclass
class CandidateResult:
    name: str
    family: str
    fold_mape: list[float] = field(default_factory=list)
    fold_rmse: list[float] = field(default_factory=list)
    error: str = ""
    forecast: Optional[np.ndarray] = None       # final future forecast
    backtest_preds: Optional[np.ndarray] = None  # last-fold preds (for stacking)
    backtest_actuals: Optional[np.ndarray] = None

    @property
    def mape(self) -> float:
        vals = [m for m in self.fold_mape if np.isfinite(m)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def rmse(self) -> float:
        vals = [m for m in self.fold_rmse if np.isfinite(m)]
        return float(np.mean(vals)) if vals else float("nan")


def prepare_series(df: pd.DataFrame, date_col: str, target_col: str,
                   preprocessing: Optional[dict] = None) -> pd.Series:
    """Aggregate to a regular series; apply Planner/Critic preprocessing."""
    ts = (df[[date_col, target_col]].dropna()
          .groupby(date_col)[target_col].sum().sort_index())
    prep = preprocessing or {}
    if prep.get("winsorize"):
        lo, hi = ts.quantile(0.01), ts.quantile(0.99)
        ts = ts.clip(lo, hi)
    if prep.get("fill_gaps"):
        freq = pd.infer_freq(ts.index) or "D"
        ts = ts.asfreq(freq).interpolate(limit=7)
    return ts


def run_backtest(ts: pd.Series, candidates: list[str], period: int = 7,
                 horizon: int = 30, n_folds: int = 3) -> dict[str, CandidateResult]:
    """Rolling-origin: k folds, each trains on prefix, predicts next horizon."""
    registry = {m.name: m for m in build_registry()}
    horizon = min(horizon, max(len(ts) // 6, 7))
    results: dict[str, CandidateResult] = {}

    fold_starts = [len(ts) - (n_folds - i) * horizon for i in range(n_folds)]
    fold_starts = [s for s in fold_starts if s > max(2 * period, 30)]

    for name in candidates:
        model = registry.get(name)
        if model is None or not model.available:
            results[name] = CandidateResult(name, model.family if model else "?",
                                            error="unavailable")
            continue
        res = CandidateResult(name, model.family)
        for start in fold_starts:
            train, test = ts.iloc[:start], ts.iloc[start:start + horizon]
            if test.empty:
                continue
            try:
                pred = model.runner(train, len(test), period=period)
                res.fold_mape.append(mape(test.to_numpy(), pred))
                res.fold_rmse.append(rmse(test.to_numpy(), pred))
                res.backtest_preds, res.backtest_actuals = pred, test.to_numpy()
            except Exception as e:  # noqa: BLE001
                res.error = f"{type(e).__name__}: {e}"
                break
        # final future forecast on full history (only if backtest worked)
        if res.fold_mape and not res.error:
            try:
                res.forecast = model.runner(ts, horizon, period=period)
            except Exception as e:  # noqa: BLE001
                res.error = f"final fit failed: {e}"
        results[name] = res
    return results


def metrics_table(results: dict[str, CandidateResult]) -> list[dict]:
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name, "family": r.family,
            "mape": round(r.mape, 2) if np.isfinite(r.mape) else None,
            "rmse": round(r.rmse, 2) if np.isfinite(r.rmse) else None,
            "folds": len(r.fold_mape), "error": r.error or None,
        })
    return sorted(rows, key=lambda x: (x["mape"] is None, x["mape"]))
