"""
sentinel/app/streamlit_app.py

The demo. Three views in one tabbed app:
1. Risk Map  — see where the disease is going to establish
2. Clinician Briefing — given a province, what should a GP screen for
3. Response Plan — given a risk threshold breach, what does the system trigger

Run locally:
    streamlit run app/streamlit_app.py

Deploy:
    Push to GitHub -> Streamlit Community Cloud auto-deploys.
    Add ANTHROPIC_API_KEY to Streamlit secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# ---------- Page config ----------

st.set_page_config(
    page_title="Sentinel — Climate-Driven Disease Early Warning",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Data loading (cached) ----------


@st.cache_data
def load_risk_surface() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "risk_surface.parquet")


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "italy_features.parquet")


# ---------- Helpers ----------


def risk_color(prob: float) -> str:
    """Map probability [0,1] to a colour. Sequential, perceptually uniform-ish."""
    if prob < 0.2:
        return "#2c7bb6"  # low — blue
    elif prob < 0.4:
        return "#abd9e9"  # low-mod
    elif prob < 0.6:
        return "#ffffbf"  # moderate — yellow
    elif prob < 0.8:
        return "#fdae61"  # high — orange
    else:
        return "#d7191c"  # very high — red


def risk_label(prob: float) -> str:
    if prob < 0.2:
        return "Very low"
    elif prob < 0.4:
        return "Low"
    elif prob < 0.6:
        return "Moderate"
    elif prob < 0.8:
        return "High"
    else:
        return "Very high"


def make_map(df_year: pd.DataFrame, pathogen: str) -> folium.Map:
    """Render the risk surface for one year × one pathogen as a Folium map."""
    sub = df_year[df_year["pathogen"] == pathogen].copy()

    m = folium.Map(
        location=[42.5, 12.5],  # centre of Italy
        zoom_start=6,
        tiles="cartodbpositron",
    )

    for _, row in sub.iterrows():
        prob = float(row["suitability_prob"])
        established = bool(row["currently_established"])

        # Marker size = log of population
        radius = 4 + (row["population_thousands"] / 500) ** 0.5
        radius = min(radius, 20)

        popup_html = f"""
        <b>{row['province_name']}</b> ({row['region']})<br>
        Suitability probability: <b>{prob:.0%}</b> ({risk_label(prob)})<br>
        Currently established: <b>{'Yes' if established else 'No'}</b><br>
        Population: {row['population_thousands']:,}k<br>
        <i>Model AUC: {row['model_auc']:.3f}</i>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['province_name']}: {prob:.0%}",
            color="#222" if established else "#888",
            weight=2 if established else 1,
            fill=True,
            fillColor=risk_color(prob),
            fillOpacity=0.85,
        ).add_to(m)

    return m


# ---------- Sidebar ----------

st.sidebar.title("🦟 Sentinel")
st.sidebar.caption("Climate-Driven Disease Early Warning")
st.sidebar.markdown(
    "A weather forecast — but for outbreaks. Predicting where climate-sensitive "
    "diseases will become locally established 6–12 months ahead."
)
st.sidebar.markdown("---")

view = st.sidebar.radio(
    "Choose view",
    ["Risk Map", "Clinician Briefing", "Response Plan", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Capgemini Tech4Positive Futures 2026 prototype. "
    "Data sources: ECDC VectorNet, ERA5-Land, CMIP6 SSP2-4.5."
)


# ---------- Load data ----------

risk_df = load_risk_surface()
features_df = load_features()


# ---------- Views ----------

if view == "Risk Map":
    st.title("Risk Map")
    st.caption(
        "Probability that a province is climatically suitable for vector establishment, "
        "based on a logistic-regression niche model trained on ECDC VectorNet "
        "occurrence data and ERA5-Land climate features."
    )

    col_path, col_year = st.columns([1, 1])
    with col_path:
        pathogen_options = {
            "aedes_albopictus": "Dengue (Aedes albopictus)",
            "ixodes_ricinus": "Lyme (Ixodes ricinus)",
        }
        pathogen = st.selectbox(
            "Pathogen / vector",
            options=list(pathogen_options.keys()),
            format_func=lambda k: pathogen_options[k],
        )
    with col_year:
        year = st.select_slider(
            "Climate scenario year",
            options=["present", "2030", "2050"],
            value="present",
        )

    df_year = risk_df[risk_df["year"] == year]

    # Top-line metrics
    sub = df_year[df_year["pathogen"] == pathogen]
    high_risk_count = (sub["suitability_prob"] >= 0.6).sum()
    pop_at_risk = sub[sub["suitability_prob"] >= 0.6]["population_thousands"].sum()
    new_provinces = sub[
        (sub["currently_established"] == 0) & (sub["suitability_prob"] >= 0.5)
    ]

    m1, m2, m3 = st.columns(3)
    m1.metric("Provinces at moderate-to-high risk", high_risk_count)
    m2.metric("Population at risk", f"{pop_at_risk:,}k")
    m3.metric(
        "Newly suitable provinces",
        len(new_provinces),
        help="Currently unestablished provinces with ≥50% projected suitability",
    )

    # Map
    fmap = make_map(df_year, pathogen)
    st_folium(fmap, use_container_width=True, height=520, returned_objects=[])

    # Highlight: provinces becoming newly suitable
    if year != "present" and len(new_provinces):
        st.warning(
            f"**Newly suitable provinces by {year}** — these are currently "
            "unestablished but project as climatically suitable under SSP2-4.5. "
            "These are the priority targets for Sentinel sensing deployment."
        )
        st.dataframe(
            new_provinces[["province_name", "region", "suitability_prob", "population_thousands"]]
            .sort_values("suitability_prob", ascending=False)
            .rename(
                columns={
                    "province_name": "Province",
                    "region": "Region",
                    "suitability_prob": "Probability",
                    "population_thousands": "Population (k)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("How this map is built"):
        st.markdown(
            """
            - **Training data:** ECDC VectorNet 2024 vector occurrence records.
            - **Features:** Mean annual temperature, summer maximum, winter minimum,
              annual precipitation, relative humidity (ERA5-Land 1991–2020 climatology).
            - **Model:** Class-balanced logistic regression with leave-one-out
              cross-validation. Simple, interpretable, defensible at small N.
            - **Future projections:** SSP2-4.5 climate deltas applied per IPCC AR6
              Atlas Mediterranean region (~0.4°C/decade warming, summer-amplified).
            - **Honest limitation:** This prototype uses 38 NUTS-3 provinces with
              synthesised climate features following published latitudinal gradients.
              Production deployment would use raw ERA5-Land 0.1° rasters from CDS.
            """
        )


elif view == "Clinician Briefing":
    st.title("Clinician Briefing")
    st.caption(
        "Designed for GPs, public health officers, and emergency department clinicians "
        "in receiving regions. Translates risk surfaces into locally-relevant guidance."
    )

    province = st.selectbox(
        "Province",
        options=sorted(features_df["province_name"].unique()),
        index=sorted(features_df["province_name"].unique()).index("L'Aquila")
        if "L'Aquila" in features_df["province_name"].values
        else 0,
    )

    prov_row = features_df[features_df["province_name"] == province].iloc[0]
    risk_rows = risk_df[risk_df["province_name"] == province]

    # Summary card
    st.markdown(f"### {prov_row['province_name']}, {prov_row['region']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Population", f"{prov_row['population_thousands']:,}k")
    c2.metric("Mean annual temp", f"{prov_row['mean_temp_c']:.1f} °C")
    c3.metric("Summer max", f"{prov_row['summer_max_c']:.1f} °C")

    st.markdown("---")

    # Risk timeline for this province
    st.markdown("#### Vector suitability timeline")
    pivot = risk_rows.pivot(
        index="pathogen_display", columns="year", values="suitability_prob"
    )[["present", "2030", "2050"]]
    st.dataframe(
        (pivot * 100).round(0).astype(int).astype(str) + "%",
        use_container_width=True,
    )

    st.markdown("#### Recommended clinical action")

    # Simple rule-based briefing — Day 5 stretch goal: replace with Claude API call
    aedes_2030 = risk_rows[
        (risk_rows["pathogen"] == "aedes_albopictus") & (risk_rows["year"] == "2030")
    ]["suitability_prob"].iloc[0]
    aedes_now = risk_rows[
        (risk_rows["pathogen"] == "aedes_albopictus") & (risk_rows["year"] == "present")
    ]["suitability_prob"].iloc[0]
    aedes_established = risk_rows[
        (risk_rows["pathogen"] == "aedes_albopictus") & (risk_rows["year"] == "present")
    ]["currently_established"].iloc[0]

    if aedes_established:
        st.error(
            "**Aedes albopictus is currently established in this province.** "
            "Maintain heightened arboviral suspicion in patients with fever of unknown "
            "origin between May and October. Order dengue NS1 antigen + IgM and "
            "chikungunya RT-PCR for compatible presentations. Notify regional surveillance."
        )
    elif aedes_2030 > 0.5:
        st.warning(
            f"**Establishment likely by 2030 ({aedes_2030:.0%} probability).** "
            "Aedes albopictus is not currently present, but climatic conditions are "
            "becoming suitable. **Action:** Begin pre-deployment training for ED and "
            "primary care staff on dengue and chikungunya recognition. Pre-position "
            "rapid diagnostic tests via regional laboratory. Target Sentinel acoustic "
            "trap deployment Q2 2027."
        )
    else:
        st.info(
            "**Vector establishment unlikely in this horizon.** "
            "Maintain routine traveller-acquired case surveillance. No special action."
        )

    if st.button("🔮 Generate full briefing with Claude (Day 5 build)"):
        st.info(
            "**TODO (Person B, Day 5):** Wire up `anthropic` SDK call here. "
            "System prompt should include the risk row, ECDC clinical guidelines, "
            "and ask Claude to produce a personalised briefing for this province. "
            "See `app/llm.py` skeleton for the structure."
        )


elif view == "Response Plan":
    st.title("Response Plan")
    st.caption(
        "When forecast probability crosses a threshold, Sentinel generates an "
        "executable response plan — supplies, training, financial trigger."
    )

    # Find the most "newly at risk" province under 2030 climate
    newly_at_risk = risk_df[
        (risk_df["pathogen"] == "aedes_albopictus")
        & (risk_df["year"] == "2030")
        & (risk_df["currently_established"] == 0)
    ].sort_values("suitability_prob", ascending=False)

    if len(newly_at_risk):
        target = newly_at_risk.iloc[0]
        prob = float(target["suitability_prob"])

        st.markdown(
            f"### Trigger event detected: **{target['province_name']}**"
        )
        st.markdown(
            f"Aedes albopictus suitability has crossed the **0.5 threshold** under "
            f"2030 SSP2-4.5 projection (current: {prob:.0%}). Population: "
            f"{target['population_thousands']:,}k."
        )

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🚚 Pre-positioning logistics")
            rdt_count = int(target["population_thousands"] * 1.5)
            st.markdown(
                f"""
                - **Dengue NS1 RDTs:** {rdt_count:,} units to regional laboratory
                - **Chikungunya RT-PCR reagent:** 50-test pack
                - **Mobile entomology unit:** 1 deployment, Q2 2027
                - **Sentinel acoustic traps:** 8 units to high-density urban zones
                """
            )

            st.markdown("#### 🏥 Workforce preparation")
            st.markdown(
                """
                - **Auto-generated clinician briefing:** distributed to 23 GPs in catchment
                - **ED training module:** 30-minute online CME, Italian
                - **Public health officer brief:** quarterly review schedule
                """
            )

        with col_b:
            st.markdown("#### 💰 Parametric financial trigger")
            st.success(
                f"**Threshold crossed: 0.5 suitability**\n\n"
                f"**Trigger value:** €{int(target['population_thousands']) * 250:,} "
                f"released to Regional Health Authority\n\n"
                "**Source:** Simulated parametric outbreak insurance contract\n\n"
                "**Time from forecast → fund release:** < 24 hours"
            )

            st.markdown("#### 📣 Community mobilisation")
            st.markdown(
                """
                - **Public risk comms:** translated, vernacular, via municipality channels
                - **Schools toolkit:** 12 schools in target area
                - **Volunteer dispatching:** Italian Red Cross, larval-source clean-ups
                """
            )

        st.markdown("---")
        st.caption(
            "All response actions are simulated for the prototype. Production "
            "integration would push these to DHIS2 (logistics), trigger actual "
            "parametric insurance smart contracts (finance), and dispatch via "
            "IFRC and MSF operational channels (community)."
        )
    else:
        st.info("No new trigger events under current scenarios.")


elif view == "About":
    st.title("About Sentinel")
    st.markdown(
        """
        **Sentinel** is a climate-driven disease early warning system, built as
        a Capgemini Tech4Positive Futures 2026 prototype.

        ### The problem
        Climate change is redistributing infectious disease across Europe faster
        than public health systems can adapt. Dengue is now endemic in southern
        France. Tiger mosquitoes breed in London. Vibrio (flesh-eating bacteria)
        has appeared in the Baltic. The first time most countries learn dengue
        has arrived is when a child dies of it.

        ### The architecture
        1. **Sensing** — low-cost acoustic mosquito traps + eDNA + clinical syndromic feeds
        2. **Forecasting** — ecological niche models on ERA5 + CMIP6 climate
        3. **Intelligence** — LLM-driven, multilingual, persona-aware risk synthesis
        4. **Delivery** — DHIS2 push, ECDC feed, clinician app, SMS gateway
        5. **Action** — pre-positioning, vector control, parametric finance triggers

        ### What this prototype demonstrates
        Layers 2, 3, and a slice of 5 — for two pathogens (Aedes albopictus
        and Ixodes ricinus) across Italian provinces. Plus a working Raspberry Pi
        acoustic classifier as the proof of layer 1.

        ### Partners
        - Wellcome Climate & Health Team
        - LSHTM Logan Group / Global Vector Hub
        - ECDC VectorNet
        - HISP Centre (DHIS2)

        ### Team
        Capgemini Financial Services UK, T4PF 2026.
        """
    )
