"""Deferment IQ — base-management / lost-oil dashboard (multipage).

Deterministic deferment accounting (potential vs actual, by reason code) with an
optional LLM reason-code classifier and an LLM-narrated VP review. Bring-your-own-key.

Multipage (``st.navigation`` + ``st.Page``): a **Fleet Overview** page (KPIs,
base-management review, recovery work-queue, classifier eval, and a sortable
fleet table) plus one **drill-down page per well** (its potential-vs-actual +
deferred bars, the well's events, KPIs, and its recovery items). Detection /
deferment / reason-code logic is unchanged; the LLM stays BYOK-optional.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from functools import partial
from pathlib import Path

# Ensure repo root + demo dir are importable so ``src.*`` and the vendored
# ``theme`` / ``fleet_registry`` resolve regardless of cwd / Streamlit context.
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
for _p in (str(REPO_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Self-heal stale bytecode / module cache (Streamlit / HF container reuse).
import shutil as _shutil
for _pyc in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pyc, ignore_errors=True)
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fleet_registry
import theme
from src import __version__
from src import analytics as A
from src.data_loader import load_events, load_fleet
from src.deferment import classify_events, compute_deferment
from src.narrator import MissingAPIKey, render_review_markdown, write_review

DATA = REPO_ROOT / "data" / "synthetic"
WELLS = DATA / "wells"
EVAL = REPO_ROOT / "evals" / "results" / "summary.json"
AFE_COPILOT_URL = "https://diazaeric1-afe-copilot.hf.space"


# ---- bootstrap + cached loads ----------------------------------------------

def _bootstrap():
    if not any(WELLS.glob("well_*.csv")):
        with st.status("First-time setup: generating synthetic fleet…", expanded=False):
            subprocess.run([sys.executable, str(DATA / "generate.py")], check=True)


@st.cache_data(show_spinner=False)
def _load(price_per_bbl, use_llm_flag, has_key, byok_key):
    """Cache the fleet load + classification + deferment compute. Keyed on the
    LLM toggle + key presence so a deterministic run (no key) caches cleanly."""
    fleet = load_fleet(WELLS)
    events = load_events(DATA / "events.csv")
    client = None
    if use_llm_flag and has_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=byok_key)
    evc = classify_events(events, use_llm=use_llm_flag and has_key, client=client)
    daily = compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


@st.cache_data(show_spinner=False)
def _fleet_well_ids() -> list[str]:
    """Sorted well ids for navigation wiring (cheap glob, no CSV parse)."""
    return sorted(p.stem for p in WELLS.glob("well_*.csv"))


# ---- shared helpers --------------------------------------------------------

def _sidebar_controls() -> tuple[float, str, bool]:
    """Render the shared sidebar settings and return (price, byok_key, use_llm)."""
    with st.sidebar:
        st.header("Settings")
        price = st.number_input("Realized oil price ($/bbl)", 20.0, 150.0, 70.0, 1.0,
                                key="oil_price")
        byok_key = st.text_input(
            "🔑 Anthropic API key (optional)", type="password", key="byok_key",
            help="Bring your own key — used only for this session, never stored. Powers the LLM "
                 "reason-code classifier and the narrated VP review. Everything else works without it.")
        use_llm = st.checkbox("🤖 Use LLM for reason-code classification", value=False,
                              key="use_llm",
                              help="Re-classify event notes with Claude (needs key). Default is the "
                                   "deterministic rules classifier.")
    return price, byok_key, use_llm


def _back_to_overview():
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to Fleet overview", icon="📊")
    except Exception:
        pass


def _build_fleet_table(daily: pd.DataFrame, price: float) -> pd.DataFrame:
    """One row per well joined with registry metadata + recovery $ / capture %.

    Reuses ``analytics.top_wells`` (deferment + dominant cause + uptime) and
    ``analytics.recovery_queue`` (recoverable $) so the numbers match the rest of
    the app; capture % = recoverable $ / deferred $ per well."""
    well_ids = sorted(daily["well_id"].unique()) if len(daily) else []
    if not well_ids:
        return pd.DataFrame()

    # top_wells over the whole fleet (n = all) → per-well deferred bbl/$, cause, uptime.
    tw = A.top_wells(daily, n=len(well_ids)).set_index("well_id")
    queue = A.recovery_queue(daily, oil_price=price)
    rec_by_well = (queue.groupby("well_id")["recoverable_usd"].sum().to_dict()
                   if len(queue) else {})

    rows = []
    for wid in well_ids:
        meta = fleet_registry.get(wid)
        deferred_bbl = float(tw.loc[wid, "deferred_bbl"]) if wid in tw.index else 0.0
        deferred_usd = float(tw.loc[wid, "deferred_usd"]) if wid in tw.index else 0.0
        cause = str(tw.loc[wid, "top_cause"]) if wid in tw.index else "—"
        uptime = float(tw.loc[wid, "uptime_pct"]) if wid in tw.index else 100.0
        rec_usd = float(rec_by_well.get(wid, 0.0))
        capture = (rec_usd / deferred_usd * 100.0) if deferred_usd > 0 else 0.0
        rows.append({
            "Well": wid,
            "Lift": meta.lift,
            "Lateral (ft)": meta.lateral_length_ft,
            "Basin · Formation": f"{meta.basin} · {meta.formation}",
            "Deferred bbl": round(deferred_bbl, 0),
            "Deferred $": round(deferred_usd, 0),
            "Dominant cause": cause,
            "Uptime %": round(uptime, 1),
            "Recoverable $": round(rec_usd, 0),
            "Capture %": round(capture, 0),
        })
    out = pd.DataFrame(rows).sort_values("Deferred $", ascending=False).reset_index(drop=True)
    return out


# =====================================================================
# PAGE: Fleet overview
# =====================================================================

def render_overview() -> None:
    price, byok_key, use_llm = _sidebar_controls()

    theme.header(
        "Deferment IQ",
        subtitle="Base management / lost-oil accounting — where are the barrels going, what's it costing, "
                 "and what's recoverable. Built by an ex-OXY / ex-Shell Staff Production Engineer.",
        chips=[(f"v{__version__}", "ver"), ("~92% reason-code acc", "eval"),
               ("fleet explorer", "info")],
    )

    with st.expander(f"🆕 What is this / v{__version__}"):
        st.markdown(
            "- **Deferment vs. potential** — each well's entitlement is modeled from its full-uptime "
            "days (P75, decline-aware); the gap to actual is split into **downtime** vs. **underperformance**.\n"
            "- **Reason-code attribution** — every lost barrel is tagged to a cause from the operator's "
            "free-text note (deterministic rules classifier, ~92% on the eval; optional LLM for the long tail).\n"
            "- **The VP views** — deferment waterfall, Pareto of $ by cause, worst-offender wells, MTTR, and "
            "the **recoverable** opportunity (excludes planned + reservoir, which you can't get back).\n"
            "- **Capture rate** flags uncaptured (un-coded) deferment — a real data-quality gap to close.\n"
            "- Deterministic engine; the LLM only classifies the messy tail and narrates. Bring your own key.\n"
            "\n"
            "**New — fleet explorer (multipage):**\n"
            "- **Fleet Overview** + a **drill-down page per well** (`st.navigation`) — open any well from the "
            "**Wells** section in the sidebar for its potential-vs-actual + deferred-bar chart, its events, "
            "and its recovery items.\n"
            "- **Sortable fleet table** — one row per well with lift, lateral, basin·formation, deferred "
            "bbl/$, dominant cause, uptime %, recoverable $, and capture %.\n"
            "- Prioritized recovery work-queue (recoverable $ ÷ MTTR), MTTR-by-cause, shared fleet registry, "
            "unified suite theme + cross-app navigator."
        )

    fleet, evc, daily = _load(price, use_llm, bool(byok_key), byok_key)
    k = A.fleet_kpis(daily, price)
    pareto = A.pareto_by_cause(daily)
    top = A.top_wells(daily, 10)
    rec = A.recovery_opportunity(daily)
    queue = A.recovery_queue(daily, evc, price)

    tab_review, tab_queue, tab_table, tab_eval = st.tabs(
        ["📋 Base-Management Review", "🔧 Recovery queue", "📋 Fleet table", "🎯 Classifier eval"])

    with tab_review:
        _review_section(k, pareto, top, rec, daily, evc, price, byok_key)
    with tab_queue:
        _queue_section(queue)
    with tab_table:
        _fleet_table_section(daily, price)
    with tab_eval:
        _eval_section()


def _review_section(k, pareto, top, rec, daily, evc, price, byok_key) -> None:
    c = st.columns(5)
    c[0].metric("Production efficiency", f"{k['uptime_pct']:.1f}%", help="Actual ÷ potential")
    c[1].metric("Deferred", f"${k['deferred_usd']:,.0f}", f"{k['pct_deferred']:.1f}% of potential",
                delta_color="inverse")
    c[2].metric("Deferred rate", f"{k['deferred_bopd_avg']:,.0f} BOPD")
    c[3].metric("Recoverable opportunity", f"${rec['recoverable_usd']:,.0f}")
    c[4].metric("Reason-code capture", f"{k['capture_rate_pct']:.0f}%",
                delta=("coding gap" if k['capture_rate_pct'] < 90 else "good"),
                delta_color=("inverse" if k['capture_rate_pct'] < 90 else "off"))

    left, right = st.columns(2)
    with left:
        st.subheader("Deferment waterfall (bbl)")
        wf = A.waterfall(daily)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(wf) - 2) + ["total"],
            x=[s["label"] for s in wf], y=[s["value"] for s in wf],
            connector={"line": {"color": theme.GREY}},
            decreasing={"marker": {"color": theme.RED}},
            increasing={"marker": {"color": theme.BLUE}},
            totals={"marker": {"color": theme.NAVY}}))
        st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")
    with right:
        st.subheader("Where the barrels go — $ by cause")
        if len(pareto):
            pf = go.Figure()
            pf.add_bar(x=pareto["label"], y=pareto["deferred_usd"], name="Deferred $",
                       marker_color=[theme.BLUE if r else theme.GREY for r in pareto["recoverable"]])
            pf.add_scatter(x=pareto["label"], y=pareto["cum_pct"], name="Cumulative %",
                           yaxis="y2", line=dict(color=theme.RED))
            pf.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="cum %"))
            st.plotly_chart(theme.style_fig(pf, height=380), width="stretch")
            st.caption("Blue = recoverable · grey = planned/reservoir (not recoverable).")

    st.subheader("Worst-offender wells")
    disp = top.copy()
    disp["deferred_usd"] = disp["deferred_usd"].map(lambda v: f"${v:,.0f}")
    disp["deferred_bbl"] = disp["deferred_bbl"].map(lambda v: f"{v:,.0f}")
    disp["uptime_pct"] = disp["uptime_pct"].map(lambda v: f"{v:.0f}%")
    disp.columns = ["Well", "Deferred bbl", "Deferred $", "Dominant cause", "Uptime"]
    st.dataframe(disp, width="stretch", hide_index=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.subheader("MTTR by cause (days)")
        m = A.mttr_by_cause(evc)
        if len(m):
            mm = m.copy(); mm["mttr_days"] = mm["mttr_days"].map(lambda v: f"{v:.1f}")
            st.dataframe(mm[["label", "n_events", "mttr_days", "total_event_days"]]
                         .rename(columns={"label": "Cause", "n_events": "Events",
                                          "mttr_days": "MTTR (d)", "total_event_days": "Down-days"}),
                         width="stretch", hide_index=True)
            ms = m.sort_values("mttr_days")
            mf = go.Figure(go.Bar(x=ms["mttr_days"], y=ms["label"], orientation="h",
                                  marker_color=theme.AMBER))
            mf.update_layout(xaxis_title="MTTR (days)")
            st.plotly_chart(theme.style_fig(mf, height=260, legend=False), width="stretch")
    with mc2:
        st.subheader("Deferment trend (weekly bbl)")
        tr = A.deferment_trend(daily, "W")
        tf = go.Figure(go.Scatter(x=tr["date"], y=tr["deferred_bbl"], fill="tozeroy",
                                  line=dict(color=theme.RED)))
        st.plotly_chart(theme.style_fig(tf, height=260, legend=False), width="stretch")

    st.divider()
    st.subheader("📝 Senior-PE base-management review")
    if st.button("Generate review", type="primary"):
        try:
            client = None
            if byok_key:
                from anthropic import Anthropic
                client = Anthropic(api_key=byok_key)
            with st.spinner("Writing the VP review…"):
                md = write_review(k, pareto, top, rec, brief_date=date.today().isoformat(), client=client)
            st.markdown(md)
        except MissingAPIKey:
            st.info("No API key — showing the deterministic review. Add your Anthropic key in the "
                    "sidebar for the Senior-PE narrated version.")
            st.markdown(render_review_markdown(k, pareto, top, rec))


def _queue_section(queue) -> None:
    st.subheader("Prioritized recovery work-queue")
    st.caption(
        "From *where are the barrels lost* to *what to do next, what it's worth, who acts* — "
        "the **Quantify → Authorize** handoff. One actionable item per (well, recoverable cause); "
        "planned work and reservoir/watering-out are excluded (you can't get those barrels back). "
        "Ranked by **priority_score = recoverable $ ÷ MTTR (days)** — value per day-to-restore, so "
        "a quick high-value win outranks a slow one of similar size.")

    if not len(queue):
        st.info("No recoverable deferment in the current period — nothing to queue.")
        return

    total_rec_usd = float(queue["recoverable_usd"].sum())
    n_items = int(len(queue))
    toprow = queue.iloc[0]

    kc = st.columns(3)
    kc[0].metric("Total recoverable", f"${total_rec_usd:,.0f}",
                 help="Sum of recoverable $ across every queued item.")
    kc[1].metric("Actionable items", f"{n_items}",
                 help="Distinct (well, recoverable cause) interventions.")
    kc[2].metric("Fastest high-value win",
                 f"{toprow['well_id']} · {toprow['cause']}",
                 f"${toprow['recoverable_usd']:,.0f} · {toprow['mttr_days']:.1f}d",
                 help="Highest value-per-day-to-restore item — do this first.")

    bar = queue.head(12).iloc[::-1]
    causes = list(dict.fromkeys(queue["cause"]))
    cmap = {c: theme.COLORWAY[i % len(theme.COLORWAY)] for i, c in enumerate(causes)}
    bf = go.Figure()
    for c in causes:
        sub = bar[bar["cause"] == c]
        if not len(sub):
            continue
        bf.add_bar(
            y=[f"{w} · {c}" for w in sub["well_id"]], x=sub["recoverable_usd"],
            name=c, orientation="h", marker_color=cmap[c],
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
    bf.update_layout(barmode="stack", xaxis_title="Recoverable $",
                     title="Top recovery opportunities by $ (colored by cause)")
    st.plotly_chart(theme.style_fig(bf, height=420), width="stretch")

    disp = queue.copy()
    disp.insert(0, "#", range(1, len(disp) + 1))
    disp["recoverable_usd"] = disp["recoverable_usd"].map(lambda v: f"${v:,.0f}")
    disp["recoverable_bbl"] = disp["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
    disp["mttr_days"] = disp["mttr_days"].map(lambda v: f"{v:.1f}")
    disp["priority_score"] = disp["priority_score"].map(lambda v: f"{v:,.0f}")
    disp = disp[["#", "well_id", "cause", "suggested_action",
                 "recoverable_bbl", "recoverable_usd", "mttr_days", "priority_score"]]
    disp.columns = ["#", "Well", "Cause", "Suggested action",
                    "Recoverable bbl", "Recoverable $", "MTTR (d)", "Priority ($/day)"]
    st.dataframe(disp, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Authorize the top interventions")
    st.caption("Each item is sized and ready to hand to capital authorization.")
    for _, r in queue.head(5).iterrows():
        st.markdown(
            f"**{r['well_id']} — {r['cause']}** · {r['suggested_action']} · "
            f"recover **{r['recoverable_bbl']:,.0f} bbl (${r['recoverable_usd']:,.0f})**, "
            f"~{r['mttr_days']:.1f}-day restore — "
            f"[authorize the intervention in AFE Copilot ↗]({AFE_COPILOT_URL})")
    st.caption("Deep-links open AFE Copilot in a new tab to draft the Authorization for Expenditure.")


def _fleet_table_section(daily, price) -> None:
    st.caption("One row per well — sort any column. Open a well from the **Wells** section in "
               "the sidebar to drill in (potential-vs-actual, deferred bars, events, recovery items).")
    table = _build_fleet_table(daily, price)
    if table.empty:
        st.info("No fleet data available.")
        return
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Deferred $": st.column_config.NumberColumn(format="$%d"),
            "Recoverable $": st.column_config.NumberColumn(format="$%d"),
            "Deferred bbl": st.column_config.NumberColumn(format="%d"),
            "Lateral (ft)": st.column_config.NumberColumn(format="%d"),
            "Uptime %": st.column_config.NumberColumn(format="%.1f%%"),
            "Capture %": st.column_config.NumberColumn(format="%d%%"),
        })


def _eval_section() -> None:
    st.subheader("Reason-code classifier — eval vs. ground-truth causes")
    st.caption("The event log carries a ground-truth cause the classifier never sees. The deterministic "
               "rules classifier is scored on it (precision/recall/F1 + accuracy). A CI gate fails the "
               "build under 80%. Run `python -m evals.run_evals` to refresh.")
    if EVAL.exists():
        res = json.loads(EVAL.read_text())
        e1, e2 = st.columns(2)
        e1.metric("Overall accuracy", f"{res['accuracy']*100:.0f}%", f"{res['n']} events")
        e2.metric("Classes", len(res["per_class"]))
        rows = [{"Cause": c, "Precision": m["precision"], "Recall": m["recall"],
                 "F1": m["f1"], "n": m["support"]} for c, m in res["per_class"].items()]
        pc = pd.DataFrame(rows)
        for col in ("Precision", "Recall", "F1"):
            pc[col] = pc[col].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—")
        st.dataframe(pc, width="stretch", hide_index=True)
        st.caption("Residual misses are the deliberately vague notes (e.g. \"well down, see foreman\") — "
                   "exactly where the optional LLM classifier earns its keep.")
    else:
        st.info("No eval summary yet — run `python -m evals.run_evals`.")


# =====================================================================
# PAGE: per-well drill-down
# =====================================================================

def render_well(well_id: str) -> None:
    price, byok_key, use_llm = _sidebar_controls()
    fleet, evc, daily = _load(price, use_llm, bool(byok_key), byok_key)
    meta = fleet_registry.get(well_id)

    theme.header(
        f"{well_id} · {meta.name}",
        subtitle=f"{meta.lift} · {meta.basin} · {meta.formation} · {meta.area}",
        chips=[(f"v{__version__}", "ver"), (meta.peer_group, "info")],
    )
    theme.well_cross_links("deferment", well_id)
    _back_to_overview()

    wd = daily[daily["well_id"] == well_id]
    if not len(wd):
        st.warning("No production history for this well.")
        _back_to_overview()
        return

    deferred_bbl = float(wd["total_def"].sum())
    deferred_usd = deferred_bbl * price
    potential = float(wd["potential"].sum())
    actual = float(wd["bopd"].sum())
    uptime = (actual / potential * 100.0) if potential > 0 else 100.0

    # dominant cause + recovery items for this well (reuse the same analytics).
    loss = wd[wd["total_def"] > 1e-6]
    if len(loss):
        cause_key = (loss.groupby("reason_key")["deferred_usd"].sum()
                     .sort_values(ascending=False).index[0])
        from src.reason_codes import label_for
        dominant_cause = label_for(cause_key)
    else:
        dominant_cause = "—"
    well_queue = A.recovery_queue(wd, evc[evc["well_id"] == well_id] if "well_id" in evc.columns
                                  else None, price)

    m = st.columns(5)
    m[0].metric("Deferred bbl", f"{deferred_bbl:,.0f}")
    m[1].metric("Deferred $", f"${deferred_usd:,.0f}", delta_color="inverse")
    m[2].metric("Uptime %", f"{uptime:.1f}%", help="Actual ÷ potential over the period")
    m[3].metric("Dominant cause", dominant_cause)
    m[4].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")

    # potential (dashed) vs actual BOPD + deferred bars overlay
    st.subheader("Potential vs. actual — deferred barrels")
    fig = go.Figure()
    fig.add_scatter(x=wd["date"], y=wd["potential"], name="Potential",
                    line=dict(color=theme.BLUE, dash="dash"))
    fig.add_scatter(x=wd["date"], y=wd["bopd"], name="Actual BOPD",
                    line=dict(color=theme.NAVY))
    fig.add_bar(x=wd["date"], y=wd["total_def"], name="Deferred",
                marker_color=theme.RED, opacity=0.5)
    st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")

    # events table for this well
    ev = evc[evc["well_id"] == well_id] if "well_id" in evc.columns else pd.DataFrame()
    if len(ev):
        st.subheader("Events for this well")
        show = ev[["start_date", "end_date", "note", "reason_key"]].copy()
        from src.reason_codes import label_for
        show["reason_key"] = show["reason_key"].map(label_for)
        show.columns = ["Start", "End", "Operator note", "Classified cause"]
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.caption("No downtime/curtailment events logged for this well.")

    # this well's recovery items
    st.subheader("Recovery items for this well")
    if len(well_queue):
        wq = well_queue.copy()
        wq["recoverable_usd"] = wq["recoverable_usd"].map(lambda v: f"${v:,.0f}")
        wq["recoverable_bbl"] = wq["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
        wq["mttr_days"] = wq["mttr_days"].map(lambda v: f"{v:.1f}")
        wq = wq[["cause", "suggested_action", "recoverable_bbl", "recoverable_usd", "mttr_days"]]
        wq.columns = ["Cause", "Suggested action", "Recoverable bbl", "Recoverable $", "MTTR (d)"]
        st.dataframe(wq, width="stretch", hide_index=True)
        st.caption(f"[Authorize an intervention in AFE Copilot ↗]({AFE_COPILOT_URL})")
    else:
        st.info("No recoverable deferment for this well — nothing to queue.")

    _back_to_overview()


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("Deferment IQ", icon="🛢️")
theme.suite_nav("deferment")
_bootstrap()

overview = st.Page(render_overview, title="Fleet overview", icon="📊", default=True)
wells = [
    st.Page(partial(render_well, wid), title=wid, url_path=wid)
    for wid in _fleet_well_ids()
]
st.navigation({"Fleet": [overview], "Wells": wells}).run()
