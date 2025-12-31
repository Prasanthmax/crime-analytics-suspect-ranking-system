import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import streamlit as st

from src.similarity_engine import SimilarityEngine
from src.suspect_ranker import SuspectRanker


# ---------------------- CONFIG & STYLE ---------------------- #

st.set_page_config(
    page_title="Crime Analytics and Suspect Ranking",
    layout="wide",
)

st.markdown(
    """
    <style>
    body { font-size: 16px; }
    div.block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }
    h1, h2, h3 { font-weight: 600; }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- HELPERS ---------------------- #

@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "processed" / "clean_cases.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_resource
def get_engines(df: pd.DataFrame):
    sim = SimilarityEngine(df)
    ranker = SuspectRanker(df)
    return sim, ranker


def render_case_card(row: pd.Series, title: str):
    """Nicely formatted case details instead of JSON/dict view."""
    st.markdown(f"**{title}**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"- **Case ID:** {int(row['dr_no'])}")
        st.markdown(f"- **Date:** {row['datetime'].date()}")
        st.markdown(f"- **Area:** {row['area_name']}")
    with col2:
        st.markdown(f"- **Crime type:** {row['crm_cd_desc']}")
        st.markdown(f"- **Weapon:** {row.get('weapon_desc', 'NA')}")
    with col3:
        age = row.get("vict_age")
        age_text = "NA" if pd.isna(age) else int(age)
        st.markdown(f"- **Victim age:** {age_text}")
        st.markdown(f"- **Victim sex:** {row.get('vict_sex', 'NA')}")


df = load_data()
sim_engine, ranker = get_engines(df)

cities = sorted(df["area_name"].dropna().unique().tolist())
weapons = sorted(df["weapon_desc"].dropna().unique().tolist())
case_ids = set(df["dr_no"].tolist())

default_city = df["area_name"].mode()[0] if len(cities) else None
default_weapon = df["weapon_desc"].mode()[0] if len(weapons) else None


# ---------------------- HEADER ---------------------- #

st.title("Crime Analytics and Suspect Ranking System")
st.write(
    "Interactive tool for exploring historical crime data, finding similar cases "
    "based on modus operandi, and ranking potential suspect cases."
)

tab_overview, tab_similar, tab_rank, tab_analytics = st.tabs(
    ["Overview", "Similar Cases", "Suspect Ranking", "Analytics"]
)


# ---------------------- OVERVIEW TAB ---------------------- #

with tab_overview:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-label">Total cases</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{len(df):,}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="metric-label">Distinct areas</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{df["area_name"].nunique()}</div>',
            unsafe_allow_html=True,
        )

    with col3:
        span = f"{df['datetime'].min().date()} to {df['datetime'].max().date()}"
        st.markdown('<div class="metric-label">Time span</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{span}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Recent identity theft cases (sample)")
    sample = (
        df[df["crm_cd_desc"] == "THEFT OF IDENTITY"]
        .sort_values("datetime", ascending=False)
        .head(10)[
            ["dr_no", "datetime", "area_name", "weapon_desc", "vict_age", "vict_sex"]
        ]
    )
    st.dataframe(sample, use_container_width=True)

    # extra quick insights
    st.markdown("### Quick insights")

    col_a, col_b = st.columns(2)

    with col_a:
        by_area = (
            df["area_name"].value_counts()
            .head(5)
            .rename("count")
            .reset_index()
            .rename(columns={"index": "area_name"})
        )
        st.markdown("Top 5 areas by crime count")
        st.bar_chart(by_area.set_index("area_name"))

    with col_b:
        by_weapon = (
            df["weapon_desc"]
            .fillna("UNKNOWN")
            .value_counts()
            .head(5)
            .rename("count")
            .reset_index()
            .rename(columns={"index": "weapon_desc"})
        )
        st.markdown("Top 5 weapons used")
        st.bar_chart(by_weapon.set_index("weapon_desc"))


# ---------------------- SIMILAR CASES TAB ---------------------- #

with tab_similar:
    st.subheader("Similar Cases")

    st.markdown("#### 1. Enter base case ID (DR_NO)")

    input_case_id = st.text_input(
        "Case ID",
        value="",
        placeholder="Type a numeric DR_NO and press Enter",
    )

    base_case_id = None
    if input_case_id.strip():
        try:
            typed = int(input_case_id.strip())
            if typed in case_ids:
                base_case_id = typed
            else:
                st.warning("Typed Case ID not found in dataset.")
        except ValueError:
            st.warning("Case ID must be numeric.")

    if base_case_id is not None:
        base_row = df[df["dr_no"] == base_case_id].iloc[0]
        render_case_card(base_row, "Base case details")

        st.markdown("---")
        st.markdown("#### 2. Filters for similar cases")

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

        st.markdown("---")
        st.markdown("#### 3. Run similarity search")

        similar_results = pd.DataFrame()

        if st.button("Find similar cases"):
            similar_results = sim_engine.get_similar_cases(
                base_case_id,
                top_k=top_k_sim,
                city=chosen_city,
                weapon=chosen_weapon,
            )

            if similar_results.empty:
                st.warning("No similar cases found for the selected filters.")
            else:
                st.success(f"{len(similar_results)} similar cases found.")

                show_cols = [
                    "dr_no",
                    "datetime",
                    "area_name",
                    "crm_cd_desc",
                    "weapon_desc",
                    "vict_age",
                    "vict_sex",
                    "similarity",
                ]
                st.dataframe(similar_results[show_cols], use_container_width=True)

                st.markdown("#### Detailed view of a similar case")
                inspect_id = st.selectbox(
                    "Select a case from the results",
                    options=similar_results["dr_no"].tolist(),
                )
                detail_row = similar_results[similar_results["dr_no"] == inspect_id].iloc[0]
                render_case_card(detail_row, "Selected similar case")
                st.markdown("**MO text:**")
                st.write(detail_row["mo_text"])

            # save for ranking tab
            if not similar_results.empty:
                st.session_state["latest_base_case_id"] = base_case_id
                st.session_state["latest_similar"] = similar_results.copy()
    else:
        st.info("Enter a valid Case ID above to see similar cases.")


# ---------------------- SUSPECT RANKING TAB ---------------------- #

with tab_rank:
    st.subheader("Suspect Ranking")

    st.write(
        "Ranks the most likely suspect cases for a given base case using text "
        "similarity, location, weapon and time proximity."
    )

    # try to reuse last base case if available
    default_base = st.session_state.get("latest_base_case_id", None)
    default_value = str(default_base) if default_base is not None else ""

    rank_input_id = st.text_input(
        "Enter Case ID (DR_NO) for ranking",
        value=default_value,
        placeholder="Type a numeric DR_NO",
        key="rank_case_input",
    )

    base_for_rank = None
    if rank_input_id.strip():
        try:
            rid = int(rank_input_id.strip())
            if rid in case_ids:
                base_for_rank = rid
            else:
                st.warning("Typed Case ID not found in dataset.")
        except ValueError:
            st.warning("Case ID must be numeric.")

    rank_top_k = st.slider("Top K ranked suspects", 5, 50, 10, step=5)

    if base_for_rank is not None:
        base_row_rank = df[df["dr_no"] == base_for_rank].iloc[0]
        render_case_card(base_row_rank, "Base case for ranking")

        if st.button("Run suspect ranking"):
            pool = sim_engine.get_similar_cases(
                base_for_rank,
                top_k=rank_top_k * 3,
                city="All",
                weapon="All",
            )

            if pool.empty:
                st.warning("No candidate cases available for ranking.")
            else:
                ranked = ranker.rank_suspects(base_for_rank, pool, top_k=rank_top_k)

                show_cols = [
                    "dr_no",
                    "datetime",
                    "area_name",
                    "crm_cd_desc",
                    "weapon_desc",
                    "vict_age",
                    "vict_sex",
                    "similarity",
                    "score",
                ]
                st.dataframe(ranked[show_cols], use_container_width=True)

                st.markdown("#### Detailed view of a ranked case")
                rid_select = st.selectbox(
                    "Select a ranked case to view details",
                    options=ranked["dr_no"].tolist(),
                )
                rrow = ranked[ranked["dr_no"] == rid_select].iloc[0]
                render_case_card(rrow, "Selected ranked case")
                st.markdown("**MO text:**")
                st.write(rrow["mo_text"])
    else:
        st.info("Enter a valid Case ID above to run suspect ranking.")


# ---------------------- ANALYTICS TAB ---------------------- #

with tab_analytics:
    st.subheader("Crime Analytics")

    st.markdown("#### Crimes per month")
    monthly = df.set_index("datetime").resample("M").size().to_frame("count")
    st.line_chart(monthly, height=260)

    st.markdown("#### Top areas by crime count")
    top_area = (
        df["area_name"]
        .value_counts()
        .sort_values(ascending=False)
        .head(10)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "area_name"})
    )
    st.bar_chart(top_area.set_index("area_name"), height=260)

    st.markdown("#### Top weapons used")
    top_weapon = (
        df["weapon_desc"]
        .fillna("UNKNOWN")
        .value_counts()
        .head(10)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "weapon_desc"})
    )
    st.bar_chart(top_weapon.set_index("weapon_desc"), height=260)

    st.markdown("#### Victim age distribution")
    age_series = df["vict_age"].dropna()
    if not age_series.empty:
        bins = [0, 18, 30, 45, 60, 80, 120]
        labels = ["0-17", "18-29", "30-44", "45-59", "60-79", "80+"]
        age_bins = pd.cut(age_series, bins=bins, labels=labels, right=False)
        age_counts = age_bins.value_counts().sort_index().to_frame("count")
        st.bar_chart(age_counts, height=260)
    else:
        st.info("Victim age data not available for histogram.")
