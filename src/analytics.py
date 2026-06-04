"""Base-management analytics over the daily deferment table — the views an asset VP
actually reviews: KPIs, the deferment waterfall (volume bridge), a Pareto of $ lost by
cause, the worst-offender wells, MTTR by cause, and the recovery opportunity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reason_codes import label_for


def fleet_kpis(daily: pd.DataFrame, price_per_bbl: float = 70.0) -> dict:
    if daily.empty:
        return {}
    pot = float(daily["potential"].sum())
    act = float(daily["bopd"].sum())
    deferred = float(daily["total_def"].sum())
    loss = daily[daily["total_def"] > 1e-6]
    captured = float(loss[loss["reason_key"] != "unclassified"]["total_def"].sum())
    n_days = daily["date"].nunique()
    return {
        "n_wells": int(daily["well_id"].nunique()),
        "period_days": int(n_days),
        "potential_bbl": pot,
        "actual_bbl": act,
        "deferred_bbl": deferred,
        "deferred_usd": deferred * float(price_per_bbl),
        "uptime_pct": (act / pot * 100.0) if pot > 0 else 100.0,   # production efficiency
        "pct_deferred": (deferred / pot * 100.0) if pot > 0 else 0.0,
        "capture_rate_pct": (captured / deferred * 100.0) if deferred > 0 else 100.0,
        "deferred_bopd_avg": deferred / n_days if n_days else 0.0,
    }


def pareto_by_cause(daily: pd.DataFrame) -> pd.DataFrame:
    loss = daily[daily["total_def"] > 1e-6]
    if loss.empty:
        return pd.DataFrame(columns=["reason_key", "label", "deferred_bbl", "deferred_usd",
                                     "pct_of_total", "cum_pct", "recoverable", "planned"])
    g = loss.groupby("reason_key").agg(
        deferred_bbl=("total_def", "sum"),
        deferred_usd=("deferred_usd", "sum"),
        recoverable=("recoverable", "max"),
        planned=("planned", "max"),
    ).sort_values("deferred_usd", ascending=False).reset_index()
    g["label"] = g["reason_key"].map(label_for)
    tot = g["deferred_usd"].sum()
    g["pct_of_total"] = g["deferred_usd"] / tot * 100.0 if tot else 0.0
    g["cum_pct"] = g["pct_of_total"].cumsum()
    return g[["reason_key", "label", "deferred_bbl", "deferred_usd",
              "pct_of_total", "cum_pct", "recoverable", "planned"]]


def top_wells(daily: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    loss = daily[daily["total_def"] > 1e-6]
    if loss.empty:
        return pd.DataFrame(columns=["well_id", "deferred_bbl", "deferred_usd", "top_cause", "uptime_pct"])
    by_well = daily.groupby("well_id").agg(
        deferred_bbl=("total_def", "sum"), deferred_usd=("deferred_usd", "sum"),
        potential=("potential", "sum"), actual=("bopd", "sum")).reset_index()
    # dominant cause per well
    cause = (loss.groupby(["well_id", "reason_key"])["deferred_usd"].sum()
             .reset_index().sort_values("deferred_usd", ascending=False)
             .drop_duplicates("well_id").set_index("well_id")["reason_key"])
    by_well["top_cause"] = by_well["well_id"].map(cause).map(lambda k: label_for(k) if isinstance(k, str) else "—")
    by_well["uptime_pct"] = np.where(by_well["potential"] > 0,
                                     by_well["actual"] / by_well["potential"] * 100.0, 100.0)
    return (by_well.sort_values("deferred_usd", ascending=False).head(n)
            [["well_id", "deferred_bbl", "deferred_usd", "top_cause", "uptime_pct"]]
            .reset_index(drop=True))


def waterfall(daily: pd.DataFrame) -> list[dict]:
    """Volume bridge: gross potential → minus each cause (planned first) → actual."""
    if daily.empty:
        return []
    pot = float(daily["potential"].sum())
    act = float(daily["bopd"].sum())
    steps = [{"label": "Gross potential", "value": pot, "kind": "total"}]
    par = pareto_by_cause(daily)
    # show planned losses first (expected), then unplanned biggest-first
    par = pd.concat([par[par["planned"]], par[~par["planned"]]])
    for _, r in par.iterrows():
        steps.append({"label": r["label"], "value": -float(r["deferred_bbl"]), "kind": "loss"})
    steps.append({"label": "Actual produced", "value": act, "kind": "total"})
    return steps


def mttr_by_cause(events_classified: pd.DataFrame) -> pd.DataFrame:
    """Mean-time-to-restore per cause from the event log (duration in days)."""
    if events_classified.empty or "reason_key" not in events_classified.columns:
        return pd.DataFrame(columns=["reason_key", "label", "n_events", "mttr_days", "total_event_days"])
    ev = events_classified.copy()
    ev["dur"] = (pd.to_datetime(ev["end_date"]) - pd.to_datetime(ev["start_date"])).dt.days + 1
    g = ev.groupby("reason_key").agg(n_events=("dur", "size"),
                                     mttr_days=("dur", "mean"),
                                     total_event_days=("dur", "sum")).reset_index()
    g["label"] = g["reason_key"].map(label_for)
    return g.sort_values("total_event_days", ascending=False)[
        ["reason_key", "label", "n_events", "mttr_days", "total_event_days"]].reset_index(drop=True)


def recovery_opportunity(daily: pd.DataFrame) -> dict:
    """Actionable opportunity = deferred $ in RECOVERABLE causes (excludes planned work
    and reservoir/watering-out, which you can't get back)."""
    loss = daily[daily["total_def"] > 1e-6]
    rec = loss[loss["recoverable"]]
    return {
        "recoverable_bbl": float(rec["total_def"].sum()),
        "recoverable_usd": float(rec["deferred_usd"].sum()),
        "unclassified_usd": float(loss[loss["reason_key"] == "unclassified"]["deferred_usd"].sum()),
    }


def deferment_trend(daily: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "deferred_bbl", "potential_bbl"])
    t = (daily.set_index("date").groupby(pd.Grouper(freq=freq))
         .agg(deferred_bbl=("total_def", "sum"), potential_bbl=("potential", "sum")).reset_index())
    return t
