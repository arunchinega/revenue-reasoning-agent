"""
stages/ingest.py — Stage 0: Ingest & Validate (deterministic, no LLM).

Hard gates fail fast; everything decided here is ledger-logged.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd

from core.state import RunState

MIN_ROWS_ANY = 10
MIN_ROWS_FORECAST = 30          # enforced later if forecast intent chosen
DATE_PARSE_THRESHOLD = 0.85     # fraction of values that must parse as dates


class IngestError(Exception):
    """Raised on hard-gate failure; message is user-facing."""


def _read_csv_robust(source: str | Path | io.BytesIO) -> pd.DataFrame:
    """Try common encodings/delimiters. Raise IngestError if unreadable."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            if isinstance(source, io.BytesIO):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc, sep=None, engine="python")
            if df.shape[1] == 1:
                # sep sniff may have failed; retry comma explicitly
                if isinstance(source, io.BytesIO):
                    source.seek(0)
                df2 = pd.read_csv(source, encoding=enc)
                if df2.shape[1] > 1:
                    df = df2
            return df
        except Exception as e:  # noqa: BLE001 — collect and continue
            last_err = e
    raise IngestError(f"Could not parse file as CSV: {last_err}")


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Columns where most values parse as dates."""
    candidates: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            candidates.append(col)
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= DATE_PARSE_THRESHOLD:
                candidates.append(col)
    return candidates


def run_ingest(state: RunState, source: str | Path | io.BytesIO,
               filename: str = "upload.csv") -> RunState:
    """Stage 0 entry point. Populates state.raw_df + state.ingest_report."""
    df = _read_csv_robust(source)

    # --- hard gates ---------------------------------------------------------
    if df.empty or len(df) < MIN_ROWS_ANY:
        raise IngestError(
            f"File has {len(df)} rows; need at least {MIN_ROWS_ANY} for any analysis."
        )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = _detect_date_columns(df)
    # numeric-looking object columns (e.g. "1,234.50") — coerce copy to check
    coercible = []
    for col in df.columns:
        if col in numeric_cols or col in date_cols:
            continue
        s = df[col].astype(str).str.replace(",", "", regex=False)
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() >= 0.9:
            coercible.append(col)
            df[col] = coerced
            numeric_cols.append(col)
    if not numeric_cols:
        raise IngestError("No numeric columns found — nothing to analyze.")

    # normalize detected date columns to datetime dtype
    for col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    state.raw_df = df
    state.ingest_report = {
        "filename": filename,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "coerced_numeric": coercible,
        "forecast_eligible": bool(date_cols) and len(df) >= MIN_ROWS_FORECAST,
        "gates_passed": True,
    }
    state.save_report("ingest_report", state.ingest_report)
    state.ledger.log(
        stage="ingest",
        agent="deterministic",
        decision=f"Accepted '{filename}': {len(df)} rows × {df.shape[1]} cols",
        reasoning=(
            f"{len(numeric_cols)} numeric col(s), {len(date_cols)} date col(s) detected"
            + (f"; coerced {coercible} to numeric" if coercible else "")
        ),
        evidence=["ingest.rows", "ingest.numeric_columns", "ingest.date_columns"],
        data={"forecast_eligible": state.ingest_report["forecast_eligible"]},
    )
    return state
