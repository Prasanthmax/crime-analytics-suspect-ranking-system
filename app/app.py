import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker


# ---------------------- CONFIG & STYLE ---------------------- #

st.set_page_config(
    page_title="Crime Analytics & Suspect Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Colour palette for charts (consistent across the app) --
CHART_COLORS = [
    "#7C3AED", "#6366F1", "#3B82F6", "#06B6D4", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6", "#14B8A6",
]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E2E8F0"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=CHART_COLORS,
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1260px;
    }

    /* Glassmorphism card wrapper */
    .glass-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }

    /* Metric overrides */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(124, 58, 237, 0.20);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.20);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #94A3B8 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E2E8F0 !important;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(124, 58, 237, 0.4);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #7C3AED;
        box-shadow: 0 0 16px rgba(124, 58, 237, 0.35);
    }

    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; }

    /* Toast-like subtle animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .stAlert { animation: fadeInUp 0.35s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- HELPERS ---------------------- #

REQUIRED_COLUMNS = [
    "dr_no", "datetime", "area_name", "crm_cd_desc",
    "weapon_desc", "vict_age", "vict_sex", "mo_text",
]


@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    """Load and validate the processed crime dataset."""
    data_path = ROOT / "data" / "processed" / "clean_cases.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Dataset not found at `{data_path}`. Please run preprocessing first.")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # Schema validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_resource
def get_engines(_df: pd.DataFrame):
    """Initialise similarity engine and suspect ranker (cached)."""
    sim = SimilarityEngine(_df)
    ranker = SuspectRanker(_df)
    return sim, ranker


def parse_case_id(raw: str, valid_ids: set) -> int | None:
    """Parse and validate a user-supplied case ID string.

    Returns the integer ID if valid, else shows a warning and returns None.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        cid = int(text)
    except ValueError:
        st.warning("⚠️ Case ID must be numeric.")
        return None
    if cid not in valid_ids:
        st.warning("⚠️ Case ID not found in the dataset.")
        return None
    return cid


def render_case_card(row: pd.Series, title: str):
    """Display a single case's key details in a styled card."""
    st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"🆔 **Case ID:** `{int(row['dr_no'])}`")
        dt_val = row["datetime"]
        date_str = dt_val.date() if pd.notna(dt_val) else "N/A"
        st.markdown(f"📅 **Date:** {date_str}")
        st.markdown(f"📍 **Area:** {row.get('area_name', 'N/A')}")
    with c2:
        st.markdown(f"🏷️ **Crime type:** {row.get('crm_cd_desc', 'N/A')}")
        st.markdown(f"🔫 **Weapon:** {row.get('weapon_desc', 'N/A')}")
    with c3:
        age = row.get("vict_age")
        age_text = "N/A" if pd.isna(age) else int(age)
        st.markdown(f"👤 **Victim age:** {age_text}")
        st.markdown(f"⚧ **Victim sex:** {row.get('vict_sex', 'N/A')}")

    st.markdown("</div>", unsafe_allow_html=True)


def plotly_fig(fig: go.Figure, height: int = 320):
    """Apply the global layout and render a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------- LOAD DATA ---------------------- #

df = load_data()
sim_engine, ranker = get_engines(df)

cities = sorted(df["area_name"].dropna().unique().tolist())
weapons = sorted(df["weapon_desc"].dropna().unique().tolist())
case_ids: set = set(df["dr_no"].tolist())

default_city = df["area_name"].mode().iloc[0] if len(cities) else None
default_weapon = df["weapon_desc"].mode().iloc[0] if len(weapons) else None


# ---------------------- HEADER ---------------------- #

st.title("🔍 Crime Analytics & Suspect Ranking")
st.caption(
    "Interactive tool for exploring historical crime data, finding similar cases "
    "based on modus operandi, and ranking potential suspect cases."
)

tab_overview, tab_similar, tab_rank, tab_analytics = st.tabs(
    ["📊 Overview", "🔗 Similar Cases", "🎯 Suspect Ranking", "📈 Analytics"]
)


# ---------------------- OVERVIEW TAB ---------------------- #

with tab_overview:
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Cases", f"{len(df):,}")
    with m2:
        st.metric("Distinct Areas", df["area_name"].nunique())
    with m3:
        st.metric("Crime Types", df["crm_cd_desc"].nunique())
    with m4:
        date_min = df["datetime"].min()
        date_max = df["datetime"].max()
        if pd.notna(date_min) and pd.notna(date_max):
            span_days = (date_max - date_min).days
            st.metric("Time Span", f"{span_days:,} days")
        else:
            st.metric("Time Span", "N/A")

    st.divider()

    st.markdown("### Recent Identity-Theft Cases")
    sample = (
        df[df["crm_cd_desc"] == "THEFT OF IDENTITY"]
        .sort_values("datetime", ascending=False)
        .head(10)[
            ["dr_no", "datetime", "area_name", "weapon_desc", "vict_age", "vict_sex"]
        ]
    )
    if sample.empty:
        st.info("No identity-theft cases found.")
    else:
        st.dataframe(
            sample,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("### Quick Insights")

    qa, qb = st.columns(2)

    with qa:
        by_area = (
            df["area_name"]
            .value_counts()
            .head(5)
            .reset_index()
        )
        by_area.columns = ["Area", "Count"]
        fig = px.bar(
            by_area, x="Area", y="Count",
            title="Top 5 Areas by Crime Count",
            color="Area",
            color_discrete_sequence=CHART_COLORS,
        )
        plotly_fig(fig, 300)

    with qb:
        by_weapon = (
            df["weapon_desc"]
            .fillna("UNKNOWN")
            .value_counts()
            .head(5)
            .reset_index()
        )
        by_weapon.columns = ["Weapon", "Count"]
        fig = px.bar(
            by_weapon, x="Weapon", y="Count",
            title="Top 5 Weapons Used",
            color="Weapon",
            color_discrete_sequence=CHART_COLORS,
        )
        plotly_fig(fig, 300)


# ---------------------- SIMILAR CASES TAB ---------------------- #

with tab_similar:
    st.subheader("🔗 Find Similar Cases")

    st.markdown("#### 1️⃣  Enter base case ID (DR_NO)")
    input_case_id = st.text_input(
        "Case ID",
        value="",
        placeholder="Type a numeric DR_NO and press Enter",
    )

    base_case_id = parse_case_id(input_case_id, case_ids)

    if base_case_id is not None:
        base_row = df[df["dr_no"] == base_case_id]
        if base_row.empty:
            st.error("Unexpected error: case row not found.")
            st.stop()
        render_case_card(base_row.iloc[0], "Base Case Details")

        st.divider()
        st.markdown("#### 2️⃣  Filters for similar cases")

        col_city, col_weapon, col_topk = st.columns([2, 2, 1])

        with col_city:
            city_options = ["All"] + cities
            default_city_index = (
                city_options.index(default_city) if default_city in city_options else 0
            )
            chosen_city = st.selectbox(
                "City / Area",
                options=city_options,
                index=default_city_index,
            )

        with col_weapon:
            weapon_options = ["All"] + weapons
            default_weapon_index = (
                weapon_options.index(default_weapon)
                if default_weapon in weapon_options
                else 0
            )
            chosen_weapon = st.selectbox(
                "Weapon",
                options=weapon_options,
                index=default_weapon_index,
            )

        with col_topk:
            top_k_sim = st.slider("Top K", 5, 50, 10, step=5)

        st.divider()
        st.markdown("#### 3️⃣  Run similarity search")

        if st.button("🔎 Find Similar Cases", type="primary"):
            with st.status("Searching for similar cases…", expanded=True) as status:
                similar_results = sim_engine.get_similar_cases(
                    base_case_id,
                    top_k=top_k_sim,
                    city=chosen_city,
                    weapon=chosen_weapon,
                )
                status.update(label="Search complete!", state="complete", expanded=False)

            if similar_results.empty:
                st.warning("No similar cases found for the selected filters.")
            else:
                st.success(f"✅ {len(similar_results)} similar case(s) found.")

                show_cols = [
                    "dr_no", "datetime", "area_name", "crm_cd_desc",
                    "weapon_desc", "vict_age", "vict_sex", "similarity",
                ]
                st.dataframe(
                    similar_results[show_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                # Similarity distribution mini-chart
                fig_sim = px.histogram(
                    similar_results, x="similarity", nbins=15,
                    title="Similarity Score Distribution",
                    color_discrete_sequence=["#7C3AED"],
                )
                plotly_fig(fig_sim, 250)

                st.markdown("#### Detailed view of a similar case")
                inspect_id = st.selectbox(
                    "Select a case from the results",
                    options=similar_results["dr_no"].tolist(),
                )
                detail_row = similar_results[similar_results["dr_no"] == inspect_id]
                if not detail_row.empty:
                    render_case_card(detail_row.iloc[0], "Selected Similar Case")
                    st.markdown("**MO text:**")
                    st.write(detail_row.iloc[0].get("mo_text", "N/A"))

                # Persist for ranking tab
                st.session_state["latest_base_case_id"] = base_case_id
                st.session_state["latest_similar"] = similar_results.copy()
    else:
        st.info("Enter a valid Case ID above to see similar cases.")


# ---------------------- SUSPECT RANKING TAB ---------------------- #

with tab_rank:
    st.subheader("🎯 Suspect Ranking")

    st.caption(
        "Ranks the most likely suspect cases for a given base case using text "
        "similarity, location, weapon, and time proximity."
    )

    # Reuse last base case if available
    default_base = st.session_state.get("latest_base_case_id", None)
    default_value = str(default_base) if default_base is not None else ""

    rank_input_id = st.text_input(
        "Enter Case ID (DR_NO) for ranking",
        value=default_value,
        placeholder="Type a numeric DR_NO",
        key="rank_case_input",
    )

    base_for_rank = parse_case_id(rank_input_id, case_ids)

    rank_top_k = st.slider("Top K ranked suspects", 5, 50, 10, step=5)

    if base_for_rank is not None:
        base_row_rank = df[df["dr_no"] == base_for_rank]
        if base_row_rank.empty:
            st.error("Unexpected error: base case row not found.")
        else:
            render_case_card(base_row_rank.iloc[0], "Base Case for Ranking")

            if st.button("⚡ Run Suspect Ranking", type="primary"):
                with st.status("Ranking suspects…", expanded=True) as status:
                    pool = sim_engine.get_similar_cases(
                        base_for_rank,
                        top_k=rank_top_k * 3,
                        city="All",
                        weapon="All",
                    )
                    status.update(label="Ranking complete!", state="complete", expanded=False)

                if pool.empty:
                    st.warning("No candidate cases available for ranking.")
                else:
                    ranked = ranker.rank_suspects(base_for_rank, pool, top_k=rank_top_k)

                    show_cols = [
                        "dr_no", "datetime", "area_name", "crm_cd_desc",
                        "weapon_desc", "vict_age", "vict_sex",
                        "similarity", "score",
                    ]
                    st.dataframe(
                        ranked[show_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Score distribution chart
                    fig_score = px.bar(
                        ranked.head(10),
                        x="dr_no", y="score",
                        title="Top Suspect Scores",
                        color="score",
                        color_continuous_scale=["#3B82F6", "#7C3AED", "#EF4444"],
                        labels={"dr_no": "Case ID", "score": "Composite Score"},
                    )
                    fig_score.update_xaxes(type="category")
                    plotly_fig(fig_score, 280)

                    st.markdown("#### Detailed view of a ranked case")
                    rid_select = st.selectbox(
                        "Select a ranked case to view details",
                        options=ranked["dr_no"].tolist(),
                    )
                    rrow = ranked[ranked["dr_no"] == rid_select]
                    if not rrow.empty:
                        render_case_card(rrow.iloc[0], "Selected Ranked Case")
                        st.markdown("**MO text:**")
                        st.write(rrow.iloc[0].get("mo_text", "N/A"))
    else:
        st.info("Enter a valid Case ID above to run suspect ranking.")


# ---------------------- ANALYTICS TAB ---------------------- #

with tab_analytics:
    st.subheader("📈 Crime Analytics Dashboard")
    st.caption("Comprehensive analytics across the full dataset.")

    # ---------- Row 1: Monthly trend ----------A
    st.markdown("#### Crimes per Month")
    monthly = df.set_index("datetime").resample("ME").size().reset_index(name="Count")
    monthly.columns = ["Month", "Count"]
    fig_monthly = px.area(
        monthly, x="Month", y="Count",
        title="Monthly Crime Trend",
        color_discrete_sequence=["#7C3AED"],
    )
    fig_monthly.update_traces(fill="tozeroy", line_shape="spline")
    plotly_fig(fig_monthly, 300)

    st.divider()

    # ---------- Row 2: Top areas + Top weapons ----------
    r2a, r2b = st.columns(2)

    with r2a:
        st.markdown("#### Top 10 Areas by Crime Count")
        top_area = df["area_name"].value_counts().head(10).reset_index()
        top_area.columns = ["Area", "Count"]
        fig_area = px.bar(
            top_area, x="Count", y="Area",
            orientation="h",
            title="Top 10 Areas",
            color="Count",
            color_continuous_scale=["#6366F1", "#7C3AED"],
        )
        fig_area.update_layout(yaxis=dict(autorange="reversed"))
        plotly_fig(fig_area, 360)

    with r2b:
        st.markdown("#### Top 10 Weapons Used")
        top_weapon = (
            df["weapon_desc"].fillna("UNKNOWN").value_counts().head(10).reset_index()
        )
        top_weapon.columns = ["Weapon", "Count"]
        fig_wpn = px.bar(
            top_weapon, x="Count", y="Weapon",
            orientation="h",
            title="Top 10 Weapons",
            color="Count",
            color_continuous_scale=["#06B6D4", "#3B82F6"],
        )
        fig_wpn.update_layout(yaxis=dict(autorange="reversed"))
        plotly_fig(fig_wpn, 360)

    st.divider()

    # ---------- Row 3: Age distribution + Victim sex ----------
    r3a, r3b = st.columns(2)

    with r3a:
        st.markdown("#### Victim Age Distribution")
        age_series = df["vict_age"].dropna()
        if not age_series.empty:
            bins = [0, 18, 30, 45, 60, 80, 120]
            labels = ["0–17", "18–29", "30–44", "45–59", "60–79", "80+"]
            age_bins = pd.cut(age_series, bins=bins, labels=labels, right=False)
            age_df = age_bins.value_counts().sort_index().reset_index()
            age_df.columns = ["Age Group", "Count"]
            fig_age = px.bar(
                age_df, x="Age Group", y="Count",
                title="Victim Age Groups",
                color="Age Group",
                color_discrete_sequence=CHART_COLORS,
            )
            plotly_fig(fig_age, 320)
        else:
            st.info("Victim age data not available.")

    with r3b:
        st.markdown("#### Victim Sex Breakdown")
        sex_df = df["vict_sex"].fillna("Unknown").value_counts().reset_index()
        sex_df.columns = ["Sex", "Count"]
        fig_sex = px.pie(
            sex_df, names="Sex", values="Count",
            title="Victim Sex Distribution",
            color_discrete_sequence=CHART_COLORS,
            hole=0.45,
        )
        fig_sex.update_traces(textinfo="percent+label")
        plotly_fig(fig_sex, 340)

    st.divider()

    # ---------- Row 4: Day-of-week + Hour-of-day ----------
    r4a, r4b = st.columns(2)

    with r4a:
        st.markdown("#### Crimes by Day of Week")
        df_valid_dt = df.dropna(subset=["datetime"])
        if not df_valid_dt.empty:
            dow = df_valid_dt["datetime"].dt.day_name().value_counts()
            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            dow = dow.reindex(day_order, fill_value=0).reset_index()
            dow.columns = ["Day", "Count"]
            fig_dow = px.bar(
                dow, x="Day", y="Count",
                title="Crime Count by Day of Week",
                color="Day",
                color_discrete_sequence=CHART_COLORS,
            )
            plotly_fig(fig_dow, 320)
        else:
            st.info("Date data not available for day-of-week analysis.")

    with r4b:
        st.markdown("#### Crimes by Hour of Day")
        if not df_valid_dt.empty:
            hour_counts = df_valid_dt["datetime"].dt.hour.value_counts().sort_index()
            hour_df = hour_counts.reset_index()
            hour_df.columns = ["Hour", "Count"]
            fig_hour = px.bar(
                hour_df, x="Hour", y="Count",
                title="Crime Count by Hour (0–23)",
                color="Count",
                color_continuous_scale=["#1E293B", "#7C3AED", "#EF4444"],
            )
            plotly_fig(fig_hour, 320)
        else:
            st.info("Date data not available for hourly analysis.")

    st.divider()

    # ---------- Row 5: Year-over-year trend ----------
    st.markdown("#### Year-over-Year Crime Trend")
    if not df_valid_dt.empty:
        df_yoy = df_valid_dt.copy()
        df_yoy["Year"] = df_yoy["datetime"].dt.year.astype(str)
        df_yoy["Month"] = df_yoy["datetime"].dt.month
        yoy = df_yoy.groupby(["Year", "Month"]).size().reset_index(name="Count")
        fig_yoy = px.line(
            yoy, x="Month", y="Count", color="Year",
            title="Crime Count by Month (Year-over-Year)",
            markers=True,
            color_discrete_sequence=CHART_COLORS,
        )
        fig_yoy.update_xaxes(
            tickvals=list(range(1, 13)),
            ticktext=[
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ],
        )
        plotly_fig(fig_yoy, 340)
    else:
        st.info("Date data not available for year-over-year analysis.")
