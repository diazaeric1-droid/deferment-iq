"""Deferment IQ — base-management / lost-oil dashboard.

Deterministic deferment accounting (potential vs actual, by reason code) with an
optional LLM reason-code classifier and an LLM-narrated VP review. Bring-your-own-key.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Self-heal stale bytecode / module cache (Streamlit / HF container reuse).
import shutil as _shutil
for _pyc in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pyc, ignore_errors=True)
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import __version__
from src import analytics as A
from src.data_loader import load_events, load_fleet
from src.deferment import classify_events, compute_deferment
from src.narrator import MissingAPIKey, render_review_markdown, write_review

st.set_page_config(page_title="Deferment IQ", page_icon="🛢️", layout="wide")
st.title(f"Deferment IQ  `v{__version__}`")
st.caption("Base management / lost-oil accounting — where are the barrels going, what's it costing, "
           "and what's recoverable. Built by an ex-OXY / ex-Shell Staff Production Engineer.")

with st.expander(f"🆕 What is this / v{__version__}"):
    st.markdown(
        "- **Deferment vs. potential** — each well's entitlement is modeled from its full-uptime "
        "days (P75, decline-aware); the gap to actual is split into **downtime** vs. **underperformance**.\n"
        "- **Reason-code attribution** — every lost barrel is tagged to a cause from the operator's "
        "free-text note (deterministic rules classifier, ~92% on the eval; optional LLM for the long tail).\n"
        "- **The VP views** — deferment waterfall, Pareto of $ by cause, worst-offender wells, MTTR, and "
        "the **recoverable** opportunity (excludes planned + reservoir, which you can't get back).\n"
        "- **Capture rate** flags uncaptured (un-coded) deferment — a real data-quality gap to close.\n"
        "- Deterministic engine; the LLM only classifies the messy tail and narrates. Bring your own key."
    )

DATA = REPO_ROOT / "data" / "synthetic"
WELLS = DATA / "wells"
EVAL = REPO_ROOT / "evals" / "results" / "summary.json"


def _bootstrap():
    if not any(WELLS.glob("well_*.csv")):
        with st.status("First-time setup: generating synthetic fleet…", expanded=False):
            subprocess.run([sys.executable, str(DATA / "generate.py")], check=True)


_bootstrap()

with st.sidebar:
    st.header("Settings")
    price = st.number_input("Realized oil price ($/bbl)", 20.0, 150.0, 70.0, 1.0)
    byok_key = st.text_input(
        "🔑 Anthropic API key (optional)", type="password",
        help="Bring your own key — used only for this session, never stored. Powers the LLM "
             "reason-code classifier and the narrated VP review. Everything else works without it.")
    use_llm = st.checkbox("🤖 Use LLM for reason-code classification", value=False,
                          help="Re-classify event notes with Claude (needs key). Default is the "
                               "deterministic rules classifier.")


@st.cache_data(show_spinner=False)
def _load(price_per_bbl, use_llm_flag, has_key):
    fleet = load_fleet(WELLS)
    events = load_events(DATA / "events.csv")
    client = None
    if use_llm_flag and has_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=byok_key)
    evc = classify_events(events, use_llm=use_llm_flag and has_key, client=client)
    daily = compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


fleet, evc, daily = _load(price, use_llm, bool(byok_key))
k = A.fleet_kpis(daily, price)
pareto = A.pareto_by_cause(daily)
top = A.top_wells(daily, 10)
rec = A.recovery_opportunity(daily)

tab_review, tab_wells, tab_eval = st.tabs(["📋 Base-Management Review", "🔧 Well drill-in", "🎯 Classifier eval"])

# ── Review tab ──────────────────────────────────────────────────────────────
with tab_review:
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
            connector={"line": {"color": "#bbb"}},
            decreasing={"marker": {"color": "#C0504D"}},
            increasing={"marker": {"color": "#4F81BD"}},
            totals={"marker": {"color": "#1F3A5F"}}))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Where the barrels go — $ by cause")
        if len(pareto):
            pf = go.Figure()
            pf.add_bar(x=pareto["label"], y=pareto["deferred_usd"], name="Deferred $",
                       marker_color=["#4F81BD" if r else "#9b9b9b" for r in pareto["recoverable"]])
            pf.add_scatter(x=pareto["label"], y=pareto["cum_pct"], name="Cumulative %",
                           yaxis="y2", line=dict(color="#C0504D"))
            pf.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                             yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="cum %"),
                             legend=dict(orientation="h"))
            st.plotly_chart(pf, use_container_width=True)
            st.caption("Blue = recoverable · grey = planned/reservoir (not recoverable).")

    st.subheader("Worst-offender wells")
    disp = top.copy()
    disp["deferred_usd"] = disp["deferred_usd"].map(lambda v: f"${v:,.0f}")
    disp["deferred_bbl"] = disp["deferred_bbl"].map(lambda v: f"{v:,.0f}")
    disp["uptime_pct"] = disp["uptime_pct"].map(lambda v: f"{v:.0f}%")
    disp.columns = ["Well", "Deferred bbl", "Deferred $", "Dominant cause", "Uptime"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.subheader("MTTR by cause (days)")
        m = A.mttr_by_cause(evc)
        if len(m):
            mm = m.copy(); mm["mttr_days"] = mm["mttr_days"].map(lambda v: f"{v:.1f}")
            st.dataframe(mm[["label", "n_events", "mttr_days", "total_event_days"]]
                         .rename(columns={"label": "Cause", "n_events": "Events",
                                          "mttr_days": "MTTR (d)", "total_event_days": "Down-days"}),
                         use_container_width=True, hide_index=True)
    with mc2:
        st.subheader("Deferment trend (weekly bbl)")
        tr = A.deferment_trend(daily, "W")
        tf = go.Figure(go.Scatter(x=tr["date"], y=tr["deferred_bbl"], fill="tozeroy",
                                  line=dict(color="#C0504D")))
        tf.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(tf, use_container_width=True)

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

# ── Well drill-in tab ───────────────────────────────────────────────────────
with tab_wells:
    wsel = st.selectbox("Inspect well", sorted(fleet.keys()))
    wd = daily[daily["well_id"] == wsel]
    wk = wd["total_def"].sum()
    st.metric(f"{wsel} — deferred", f"{wk:,.0f} bbl  (${wk * price:,.0f})")
    fig = go.Figure()
    fig.add_scatter(x=wd["date"], y=wd["potential"], name="Potential", line=dict(color="#4F81BD", dash="dash"))
    fig.add_scatter(x=wd["date"], y=wd["bopd"], name="Actual BOPD", line=dict(color="#1F3A5F"))
    fig.add_bar(x=wd["date"], y=wd["total_def"], name="Deferred", marker_color="#C0504D", opacity=0.5)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    ev = evc[evc["well_id"] == wsel] if "well_id" in evc.columns else pd.DataFrame()
    if len(ev):
        st.subheader("Events for this well")
        show = ev[["start_date", "end_date", "note", "reason_key"]].copy()
        show.columns = ["Start", "End", "Operator note", "Classified cause"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ── Eval tab ────────────────────────────────────────────────────────────────
with tab_eval:
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
        st.dataframe(pc, use_container_width=True, hide_index=True)
        st.caption("Residual misses are the deliberately vague notes (e.g. \"well down, see foreman\") — "
                   "exactly where the optional LLM classifier earns its keep.")
    else:
        st.info("No eval summary yet — run `python -m evals.run_evals`.")
