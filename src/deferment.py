"""Deferment engine: per-day lost-oil decomposition + reason-code attribution.

For each well-day we split the gap between potential and actual into:
  - downtime deferment : potential * (1 - runtime_fraction)   (well was OFF part of the day)
  - rate deferment     : the rest                              (UNDERPERFORMED while up — choked,
                                                                high line pressure, watering out)
These sum exactly to max(0, potential - actual). Each day's loss is attributed to the
cause of the downtime/curtailment EVENT covering that day (classified from its note);
days with a loss but no event become 'unclassified' — uncaptured deferment, itself a
finding the asset team should chase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .potential import well_potential
from .reason_codes import classify, is_planned, is_recoverable, label_for

# Losses smaller than this fraction of potential are measurement/normal-variation noise,
# not deferment — so a healthy well reads ~0 deferred (avoids phantom background loss).
DEADBAND_FRAC = 0.08


def _well_deferment(well_id: str, prod: pd.DataFrame) -> pd.DataFrame:
    pot = well_potential(prod).to_numpy(dtype=float)
    r = (prod["runtime_pct"].clip(lower=0, upper=100) / 100.0).to_numpy(dtype=float)
    bopd = prod["bopd"].to_numpy(dtype=float)

    gap = np.maximum(pot - bopd, 0.0)
    counts = gap > (DEADBAND_FRAC * pot)            # deadband: ignore within-noise gaps
    total = np.where(counts, gap, 0.0)
    downtime = np.minimum(pot * (1.0 - r), total)
    rate = total - downtime

    out = pd.DataFrame({
        "well_id": well_id,
        "date": prod["date"].values,
        "bopd": bopd,
        "runtime_pct": prod["runtime_pct"].to_numpy(dtype=float),
        "potential": pot,
        "downtime_def": downtime,
        "rate_def": rate,
        "total_def": total,
    })
    return out


def _attribution_lookup(events: pd.DataFrame, use_llm: bool, client, model: str) -> pd.DataFrame:
    """Classify each event's note once; return events with a reason_key column."""
    if events.empty:
        return events.assign(reason_key=pd.Series(dtype=str))
    ev = events.copy()
    ev["reason_key"] = [classify(n, use_llm=use_llm, client=client, model=model)
                        for n in ev["note"].fillna("")]
    return ev


def classify_events(events: pd.DataFrame, use_llm: bool = False, client=None,
                    model: str = "claude-sonnet-4-6") -> pd.DataFrame:
    """Public: classify each event's note once (adds a ``reason_key`` column).
    Pass the result to ``compute_deferment`` + ``mttr_by_cause`` to avoid re-classifying."""
    return _attribution_lookup(events, use_llm, client, model)


def _reason_for_day(well_id: str, date, ev_by_well: dict) -> str:
    for start, end, key in ev_by_well.get(well_id, ()):  # small per-well interval list
        if start <= date <= end:
            return key
    return "unclassified"


def compute_deferment(fleet: dict[str, pd.DataFrame], events: pd.DataFrame,
                      price_per_bbl: float = 70.0, use_llm: bool = False,
                      client=None, model: str = "claude-sonnet-4-6") -> pd.DataFrame:
    """Daily deferment table for the whole fleet, attributed + priced.

    Returns one row per well-day with potential, the downtime/rate split, the assigned
    reason code (+ label, recoverable, planned flags), and deferred $.
    """
    ev = events if "reason_key" in events.columns else _attribution_lookup(events, use_llm, client, model)
    ev_by_well: dict[str, list] = {}
    for _, row in ev.iterrows():
        ev_by_well.setdefault(row["well_id"], []).append(
            (row["start_date"], row["end_date"], row["reason_key"]))

    frames = [_well_deferment(wid, prod) for wid, prod in fleet.items()]
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily.empty:
        return daily

    has_loss = daily["total_def"] > 1e-6
    keys = np.where(
        has_loss.to_numpy(),
        [_reason_for_day(w, d, ev_by_well) for w, d in zip(daily["well_id"], daily["date"])],
        "",  # no loss -> no reason
    )
    daily["reason_key"] = keys
    daily["reason_label"] = [label_for(k) if k else "" for k in keys]
    daily["recoverable"] = [bool(k) and is_recoverable(k) for k in keys]
    daily["planned"] = [bool(k) and is_planned(k) for k in keys]
    daily["deferred_usd"] = daily["total_def"] * float(price_per_bbl)
    return daily
