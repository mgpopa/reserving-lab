import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reserving.io import load_triangle_csv
from reserving.triangle import to_matrix, cumulative_to_incremental
from reserving.chainladder import (
    volume_weighted_link_ratios,
    project_cumulative,
    ultimates_and_ibnr,
)
from reserving.mack import mack_se_ibnr
from reserving.bootstrap import bootstrap_total_ibnr
from reserving.scenarios import apply_inflation_shock, run_chainladder
from reserving.report import reserve_summary_template

# realistic data (CAS LRDB / Schedule P)
try:
    from trikit import load_lrdb, get_lrdb_lobs
    TRIKIT_OK = True
except Exception:
    TRIKIT_OK = False

# explanation helper
def explain(title: str, short: str, details: list[str] | None = None):
    """
    Subtle explanations:
      - one short caption always visible (if enabled)
      - optional details hidden in an expander
    """
    if not st.session_state.get("show_explanations", True):
        return
    if short:
        st.caption(short)
    if details:
        with st.expander(title, expanded=False):
            for d in details:
                st.markdown(f"- {d}")

st.set_page_config(page_title="Reserving Lab", layout="wide")

st.title("Reserving Lab: Chain Ladder, Uncertainty, Bootstrap, Scenarios")

# sidebar controls
with st.sidebar:
    st.header("Data")

    data_source_options = ["Sample (CSV)", "Upload CSV"]
    if TRIKIT_OK:
        data_source_options.append("CAS LRDB (Schedule P)")

    data_source = st.selectbox("Data source", data_source_options, index=0)

    uploaded = None
    sample_name = "ukmotor_cumulative.csv"

    if data_source == "Sample (CSV)":
        sample_name = st.selectbox(
            "Sample dataset",
            ["ukmotor_cumulative.csv"],
            index=0,
        )

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload triangle CSV", type=["csv"])

    if data_source == "CAS LRDB (Schedule P)" and TRIKIT_OK:
        st.caption("CAS Loss Reserving Database (Schedule P / NAIC).")
        lob = st.selectbox("Line of business (LOB)", get_lrdb_lobs(), index=0)
        loss_type = st.selectbox("Loss type", ["incurred", "paid"], index=0)
        train_only = st.checkbox("Training triangle only (upper-left)", value=True)
        grcode = st.number_input("Company/group code (grcode)", value=1767, step=1)

    st.divider()
    tail_factor = st.slider("Tail factor", 1.00, 1.20, 1.05, 0.01)

    st.divider()
    st.header("Bootstrap")
    n_sims = st.selectbox("Simulations", [500, 1000, 2000, 5000], index=2)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.header("UI")
    st.session_state["show_explanations"] = st.checkbox("Show explanations", value=True)

# load data into df in a consistent triangle format
if data_source == "Sample (CSV)":
    df = load_triangle_csv(f"data/{sample_name}")

elif data_source == "Upload CSV":
    if uploaded is None:
        st.warning("Upload a CSV triangle to continue.")
        st.stop()
    df = load_triangle_csv(uploaded)

elif data_source == "CAS LRDB (Schedule P)":
    if not TRIKIT_OK:
        st.error("trikit is not installed. Add trikit==0.3.3 to requirements.txt and reinstall.")
        st.stop()

    # trikit returns a triangle-like DataFrame: index=AY, columns=dev (often 1..10)
    tri = load_lrdb(
        tri_type="cum",
        lob=lob,
        loss_type=loss_type,
        grcode=int(grcode),
        train_only=bool(train_only),
    )

    df = tri.copy()
    df.insert(0, "AY", df.index.astype(str))
    df = df.reset_index(drop=True)

else:
    st.error("Unknown data source.")
    st.stop()

# small banner showing what's loaded
st.caption(f"Data source: {data_source}")
if data_source == "CAS LRDB (Schedule P)":
    st.caption(f"LOB: {lob} | loss_type: {loss_type} | grcode: {int(grcode)} | train_only: {train_only}")

# convert to internal matrix representation
ay, dev, cum = to_matrix(df)

# tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Triangle", "Chain Ladder", "Mack", "Bootstrap", "Scenarios & Summary"]
)

# Tab 1: Triangle
with tab1:
    st.subheader("Triangle (cumulative)")
    explain(
        "What this shows",
        "A cumulative claims triangle: rows are accident years (when losses happened) and columns are development ages (how claims grow over time).",
        [
            "Values are running totals up to each development age.",
            "Blank cells are future development that hasn't happened yet.",
            "This is the standard input format for Chain Ladder projections.",
        ],
    )
    st.dataframe(df, use_container_width=True)

    inc = cumulative_to_incremental(cum)
    st.subheader("Triangle (incremental)")
    explain(
        "What this shows",
        "Incremental view: the amount added during each development period (not the running total).",
        [
            "Useful for diagnosing volatility and for bootstrap simulations.",
            "Incremental values can vary more from period to period than cumulative totals.",
        ],
    )
    inc_df = pd.DataFrame(inc, columns=[str(d) for d in dev])
    inc_df.insert(0, "AY", ay)
    st.dataframe(inc_df, use_container_width=True)

    st.subheader("Heatmap (cumulative)")
    explain(
        "What this shows",
        "A quick visual summary of magnitude and maturity across accident years and development ages.",
        [
            "Older accident years typically have more observed development (more filled cells).",
            "Large values concentrated in certain areas may indicate slow/fast development or large-loss years.",
        ],
    )
    fig, ax = plt.subplots()
    ax.imshow(np.nan_to_num(cum, nan=0.0), aspect="auto")
    ax.set_yticks(range(len(ay)))
    ax.set_yticklabels(ay)
    ax.set_xticks(range(len(dev)))
    ax.set_xticklabels(dev)
    ax.set_xlabel("Development age")
    ax.set_ylabel("Accident year")
    st.pyplot(fig)

# tab 2: Chain Ladder
with tab2:
    st.subheader("Chain Ladder")

    explain(
        "What this shows",
        "Projects each accident year to an expected ultimate using historical development patterns (link ratios).",
        [
            "Link ratios estimate how cumulative claims grow from one development age to the next.",
            "Missing cells are projected forward to reach an ultimate; IBNR is the remaining amount vs the latest observed.",
        ],
    )

    link = volume_weighted_link_ratios(cum)

    st.markdown("### Link ratios")
    lr_df = pd.DataFrame({"Dev age": dev[:-1], "Link ratio to next": link})
    st.dataframe(lr_df, use_container_width=True)

    explain(
        "What this shows",
        "Average growth factors between development ages (e.g., 12→24 months).",
        [
            "Volume-weighted means accident years with larger volumes influence the factor more.",
            "Ratios closer to 1.00 at later ages usually indicate the portfolio is nearing maturity.",
        ],
    )

    st.markdown("### Ultimates and IBNR")
    proj = project_cumulative(cum, link, tail_factor=tail_factor)
    ult, ibnr = ultimates_and_ibnr(cum, proj)

    out = pd.DataFrame({"AY": ay, "Ultimate": ult, "IBNR": ibnr})
    st.dataframe(out, use_container_width=True)

    explain(
        "What this shows",
        "Ultimate is the expected final total per accident year; IBNR is what remains to develop from today's latest observed cumulative value.",
        [
            "In this app: IBNR = Ultimate − Latest observed cumulative.",
            "Newer accident years often show higher IBNR because less development has been observed.",
        ],
    )

    st.metric("Total IBNR", f"{np.nansum(ibnr):,.0f}")

# tab 3: Mack
with tab3:
    st.subheader("Mack-style uncertainty (practical approximation)")
    explain(
        "What this shows",
        "Adds an uncertainty view around the reserve estimate: standard error (SE) and coefficient of variation (CV).",
        [
            "SE(IBNR) is an absolute uncertainty measure.",
            "CV(IBNR) is relative uncertainty: SE divided by the magnitude of IBNR.",
            "Recent accident years usually have higher CV because there is less observed development.",
        ],
    )

    link = volume_weighted_link_ratios(cum)
    proj = project_cumulative(cum, link, tail_factor=tail_factor)
    ult, ibnr = ultimates_and_ibnr(cum, proj)

    se_ibnr, cv_ibnr = mack_se_ibnr(cum, link, tail_factor=tail_factor)
    mdf = pd.DataFrame({"AY": ay, "IBNR": ibnr, "SE(IBNR)": se_ibnr, "CV(IBNR)": cv_ibnr})
    st.dataframe(mdf, use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(ay, np.nan_to_num(cv_ibnr, nan=0.0))
    ax.set_title("CV(IBNR) by Accident Year")
    ax.set_ylabel("CV")
    explain(
        "What this shows",
        "Highlights which accident years have the most uncertainty relative to their reserve size.",
        [
            "This is often more informative than absolute SE when comparing years with different reserve sizes.",
        ],
    )
    st.pyplot(fig)

# Tab 4: Bootstrap
with tab4:
    st.subheader("Bootstrap reserve distribution (total IBNR)")
    explain(
        "What this shows",
        "Simulates many plausible futures to produce a distribution of total IBNR (not just a single best estimate).",
        [
            "Percentiles (P50, P90, P99) provide a risk view of reserve outcomes.",
            "VaR(99%) is the 99th percentile; TVaR(99%) is the average of the worst 1% outcomes.",
            "A wider distribution indicates higher reserve uncertainty.",
        ],
    )

    with st.spinner("Running bootstrap..."):
        dist = bootstrap_total_ibnr(cum, n_sims=int(n_sims), seed=int(seed), tail_factor=tail_factor)

    p50 = np.percentile(dist, 50)
    p90 = np.percentile(dist, 90)
    p99 = np.percentile(dist, 99)
    tvar99 = dist[dist >= p99].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P50", f"{p50:,.0f}")
    c2.metric("P90", f"{p90:,.0f}")
    c3.metric("P99 (VaR 99%)", f"{p99:,.0f}")
    c4.metric("TVaR(99%)", f"{tvar99:,.0f}")

    fig, ax = plt.subplots()
    ax.hist(dist, bins=40)
    ax.axvline(p90, linestyle="--")
    ax.axvline(p99, linestyle="--")
    ax.set_title("Bootstrap distribution of Total IBNR")
    ax.set_xlabel("Total IBNR")
    ax.set_ylabel("Frequency")
    explain(
        "What this shows",
        "Histogram of simulated total IBNR outcomes; dashed lines mark high quantiles used for a risk perspective.",
        [
            "A long right tail suggests tail risk (rare but large reserve outcomes).",
        ],
    )
    st.pyplot(fig)

# tab 5: Scenarios & Summary
with tab5:
    st.subheader("Scenario lab")
    explain(
        "What this shows",
        "Compares a base estimate against simple shocks to illustrate sensitivity to assumptions.",
        [
            "This is a stress-test tool, not a prediction of what will happen.",
            "The goal is to understand how reserves respond to key assumptions (like inflation or tail).",
        ],
    )

    shock_pct = st.slider("Inflation shock on latest observed (proxy)", 0.0, 0.30, 0.10, 0.01)

    base = run_chainladder(cum, tail_factor=tail_factor)
    shocked_cum = apply_inflation_shock(cum, shock_pct=shock_pct)
    shocked = run_chainladder(shocked_cum, tail_factor=tail_factor)

    base_total = float(np.nansum(base["ibnr"]))
    shock_total = float(np.nansum(shocked["ibnr"]))
    st.metric("Base Total IBNR", f"{base_total:,.0f}")
    st.metric("Shocked Total IBNR", f"{shock_total:,.0f}")
    st.metric("Delta", f"{(shock_total - base_total):,.0f}")

    st.subheader("Reserve summary template (to be adjusted)")
    explain(
        "What this shows",
        "A short narrative template that you can edit; numbers are pulled directly from the results above.",
        [
            "Use it to communicate the best estimate, the main drivers, and the key limitations.",
            "Keep assumptions explicit (development stability, tail factor choice, inflation handling).",
        ],
    )

    link = base["link"]
    se_ibnr, cv_ibnr = mack_se_ibnr(cum, link, tail_factor=tail_factor)

    # quick dist for template (can be slow, so I run a smaller sim and use it just for the template numbers)
    dist_for_template = bootstrap_total_ibnr(cum, n_sims=1000, seed=int(seed), tail_factor=tail_factor)

    st.text_area(
        "Draft",
        reserve_summary_template(ay, base["ult"], base["ibnr"], se_ibnr, cv_ibnr, dist_for_template),
        height=280,
    )