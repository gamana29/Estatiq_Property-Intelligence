
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
RESULTS_PATH = BASE_DIR / "results" / "model_comparison.csv"

st.set_page_config(
    page_title="Estatiq | Property Intelligence",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# PREMIUM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #111b2e;
        --panel2: #162238;
        --border: #26344c;
        --text: #edf3ff;
        --muted: #93a4bd;
        --green: #45e0a0;
        --green-dark: #173f38;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, #142b3b 0%, transparent 36%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.8);
    }

    [data-testid="stSidebar"] {
        background: #0d1627;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--text);
        margin-bottom: 0;
    }

    .brand span {
        color: var(--green);
    }

    .brand-sub {
        color: var(--muted);
        font-size: 0.77rem;
        margin-top: 3px;
        margin-bottom: 2rem;
    }

    .eyebrow {
        color: var(--green);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 3.5vw, 3.1rem);
        line-height: 1.12;
        letter-spacing: -1.7px;
        font-weight: 800;
        margin: 0;
        color: var(--text);
    }

    .hero-sub {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.7rem;
        line-height: 1.7;
    }

    .panel {
        background: linear-gradient(145deg, #142036, #101a2c);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--text);
        margin-bottom: 0.25rem;
    }

    .panel-sub {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #15233a, #101a2b);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        min-height: 125px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.7rem;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: 1.7rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: var(--text);
    }

    .metric-note {
        color: var(--green);
        font-size: 0.76rem;
        margin-top: 0.55rem;
    }

    .price-card {
        background:
            radial-gradient(circle at 100% 0%, #1b554b 0%, transparent 48%),
            linear-gradient(140deg, #142d37, #101b2d);
        border: 1px solid #2c655b;
        border-radius: 22px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }

    .price-label {
        color: #a9cfc6;
        font-size: 0.9rem;
    }

    .price-value {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.4rem, 4vw, 3.5rem);
        letter-spacing: -2px;
        line-height: 1.2;
        font-weight: 800;
        color: #f3fff9;
        margin: 0.5rem 0;
    }

    .price-note {
        color: #b7d7ce;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    div.stButton > button[kind="primary"] {
        background: var(--green);
        color: #06241c;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        padding: 0.7rem 1rem;
        min-height: 3rem;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #76efbd;
        border: none;
        color: #06241c;
        transform: translateY(-1px);
    }

    div.stButton > button {
        border-radius: 10px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #0d1728;
        border-color: #31415a;
        border-radius: 9px;
    }

    .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stSlider label {
        color: #bdc9da !important;
        font-size: 0.84rem !important;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
    }

    p, li {
        color: #c3cfe0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .status {
        display: inline-block;
        border: 1px solid #286c57;
        background: #13352e;
        color: #72edb8;
        padding: 6px 11px;
        border-radius: 100px;
        font-size: 0.76rem;
        font-weight: 700;
    }

    @media (max-width: 768px) {
        .hero-title { font-size: 2.1rem; }
        .price-value { font-size: 2.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# DATA AND MODEL LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def get_model_features(model):
    """Get the original feature columns expected by the trained pipeline."""
    preprocessor = model.named_steps.get("preprocessor")

    if preprocessor is None:
        raise ValueError(
            "The saved model does not contain a 'preprocessor' step. "
            "Check the pipeline in src/train_model.py."
        )

    features = []

    for _, _, columns in preprocessor.transformers_:
        if isinstance(columns, (list, tuple)):
            features.extend(columns)
        elif hasattr(columns, "tolist"):
            features.extend(columns.tolist())

    return list(dict.fromkeys(features))


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_chart_theme(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#c3cfe0"),
        margin=dict(l=12, r=12, t=25, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c3cfe0"),
        ),
        xaxis=dict(
            gridcolor="#26344c",
            zerolinecolor="#26344c",
            title_font=dict(color="#a9b8cc"),
        ),
        yaxis=dict(
            gridcolor="#26344c",
            zerolinecolor="#26344c",
            title_font=dict(color="#a9b8cc"),
        ),
    )
    return fig


# --------------------------------------------------
# VALIDATE FILES
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place your Ames housing dataset at data/train.csv.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Your trained model has not been saved yet.")
    st.code("python .\\src\\train_model.py", language="powershell")
    st.info("Run model training first, then refresh this page.")
    st.stop()

try:
    data = load_data()
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.stop()

if "SalePrice" not in data.columns:
    st.error("The dataset must contain a SalePrice column.")
    st.stop()

target = pd.to_numeric(data["SalePrice"], errors="coerce")
valid_data = data.loc[target.notna()].copy()
valid_data["SalePrice"] = target.loc[target.notna()]

try:
    feature_columns = get_model_features(model)
except Exception as exc:
    st.error(str(exc))
    st.stop()

missing_columns = [c for c in feature_columns if c not in valid_data.columns]

if missing_columns:
    st.error(
        "The saved model expects columns that are missing from data/train.csv: "
        + ", ".join(missing_columns[:12])
    )
    st.stop()

# Identify common Ames Housing features.
def find_column(*candidates):
    lower_map = {str(c).lower(): c for c in valid_data.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


quality_col = find_column("Overall Qual")
area_col = find_column("Gr Liv Area")
bedrooms_col = find_column("Bedroom AbvGr")
bathrooms_col = find_column("Full Bath")
garage_col = find_column("Garage Cars")
year_col = find_column("Year Built")
neighborhood_col = find_column("Neighborhood")
basement_col = find_column("Total Bsmt SF")
lot_col = find_column("Lot Area")

median_price = valid_data["SalePrice"].median()
mean_price = valid_data["SalePrice"].mean()
record_count = len(valid_data)

# Load real model comparison results if available.
comparison = None
if RESULTS_PATH.exists():
    try:
        comparison = pd.read_csv(RESULTS_PATH)
    except Exception:
        comparison = None

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">estat<span>iq</span></div>
        <div class="brand-sub">PROPERTY INTELLIGENCE PLATFORM</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Price Estimator",
            "Market Insights",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#93a4bd;font-size:0.78rem;line-height:1.8">
            <b style="color:#edf3ff">PROJECT STATUS</b><br>
            <span style="color:#45e0a0">●</span> Dataset loaded<br>
            <span style="color:#45e0a0">●</span> Model file loaded<br>
            <span style="color:#45e0a0">●</span> Prediction engine ready
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Built with Python · Scikit-learn · Streamlit")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.markdown('<div class="eyebrow">REAL ESTATE ANALYTICS / 01</div>',
                unsafe_allow_html=True)

    left, right = st.columns([4, 1])

    with left:
        st.markdown(
            '<div class="hero-title">Property intelligence,<br>powered by machine learning.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-sub">Explore housing data, estimate property values, '
            'and understand the model behind every prediction.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div style="text-align:right;padding-top:10px">'
            '<span class="status">● MODEL ONLINE</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("PROPERTIES ANALYZED", f"{record_count:,}", "Ames housing dataset")
    with c2:
        metric_card("MEDIAN SALE PRICE", money(median_price), "Observed dataset median")
    with c3:
        metric_card("AVERAGE SALE PRICE", money(mean_price), "Observed dataset average")
    with c4:
        metric_card("PROPERTY FEATURES", f"{len(feature_columns)}", "Features used by the model")

    st.write("")
    st.markdown(
        '<div class="panel-title">Market snapshot</div>'
        '<div class="panel-sub">A quick look at the housing data used by this project.</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns([1.25, 1])

    with chart_left:
        st.markdown(
            '<div class="panel-title">Sale price distribution</div>'
            '<div class="panel-sub">How recorded property prices are distributed.</div>',
            unsafe_allow_html=True,
        )
        fig = px.histogram(
            valid_data,
            x="SalePrice",
            nbins=40,
            labels={"SalePrice": "Sale price (USD)", "count": "Properties"},
            color_discrete_sequence=["#45e0a0"],
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(make_chart_theme(fig), use_container_width=True)

    with chart_right:
        st.markdown(
            '<div class="panel-title">Price by overall quality</div>'
            '<div class="panel-sub">Observed prices across quality ratings.</div>',
            unsafe_allow_html=True,
        )

        if quality_col:
            quality_data = valid_data.copy()
            quality_data[quality_col] = pd.to_numeric(
                quality_data[quality_col], errors="coerce"
            )
            fig = px.box(
                quality_data.dropna(subset=[quality_col]),
                x=quality_col,
                y="SalePrice",
                labels={
                    quality_col: "Overall quality rating",
                    "SalePrice": "Sale price (USD)",
                },
                color_discrete_sequence=["#45e0a0"],
            )
            fig.update_traces(
                marker_color="#45e0a0",
                line_color="#45e0a0",
                fillcolor="rgba(69,224,160,0.18)",
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
        else:
            st.info("No overall-quality feature was found in this dataset.")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Your workspace</div>
            <div class="panel-sub">Choose a tool to continue.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("**01 · Price Estimator**")
        st.write("Configure a property and generate a model-based price estimate.")
        if st.button("Open estimator", use_container_width=True):
            st.session_state["navigation_target"] = "Price Estimator"
            st.rerun()

    with b:
        st.markdown("**02 · Market Insights**")
        st.write("Inspect prices, quality ratings, living area, and neighborhoods.")
        if st.button("Explore market", use_container_width=True):
            st.session_state["navigation_target"] = "Market Insights"
            st.rerun()

    with c:
        st.markdown("**03 · Model Performance**")
        st.write("Review the saved evaluation metrics from model training.")
        if st.button("View performance", use_container_width=True):
            st.session_state["navigation_target"] = "Model Performance"
            st.rerun()

# --------------------------------------------------
# PRICE ESTIMATOR
# --------------------------------------------------

elif page == "Price Estimator":
    st.markdown('<div class="eyebrow">VALUATION WORKSPACE / 02</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Estimate a property\'s value.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Adjust the property details below. '
        'The trained model uses the complete feature set for its prediction.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")

    # Use a real training row as a complete starting point so every
    # feature expected by the saved pipeline is supplied.
    baseline = valid_data[feature_columns].copy()

    base_values = {}
    for col in feature_columns:
        series = baseline[col]
        if pd.api.types.is_numeric_dtype(series):
            base_values[col] = series.median()
        else:
            modes = series.mode(dropna=True)
            base_values[col] = modes.iloc[0] if not modes.empty else "Unknown"

    # A valid row of original model features.
    input_values = dict(base_values)

    with left:
        st.markdown(
            '<div class="panel-title">Configure your property</div>'
            '<div class="panel-sub">Property characteristics</div>',
            unsafe_allow_html=True,
        )

        with st.form("property_form"):
            col_a, col_b = st.columns(2)

            if quality_col:
                quality_values = pd.to_numeric(
                    valid_data[quality_col], errors="coerce"
                ).dropna()
                q_min = int(quality_values.min())
                q_max = int(quality_values.max())
                q_default = int(pd.to_numeric(
                    pd.Series([base_values[quality_col]]), errors="coerce"
                ).fillna(quality_values.median()).iloc[0])

                with col_a:
                    chosen_quality = st.slider(
                        "Overall quality",
                        min_value=q_min,
                        max_value=q_max,
                        value=max(q_min, min(q_default, q_max)),
                        help="Higher ratings indicate better overall material and finish.",
                    )
                input_values[quality_col] = chosen_quality

            if area_col:
                area_series = pd.to_numeric(valid_data[area_col], errors="coerce").dropna()
                area_default = int(area_series.median())
                with col_b:
                    chosen_area = st.number_input(
                        "Living area (sq ft)",
                        min_value=200,
                        max_value=15000,
                        value=max(200, min(area_default, 15000)),
                        step=50,
                    )
                input_values[area_col] = chosen_area

            if bedrooms_col:
                bedroom_series = pd.to_numeric(
                    valid_data[bedrooms_col], errors="coerce"
                ).dropna()
                with col_a:
                    chosen_bedrooms = st.selectbox(
                        "Bedrooms",
                        sorted(set(int(x) for x in bedroom_series if x >= 0)),
                        index=min(
                            2,
                            max(0, len(sorted(set(
                                int(x) for x in bedroom_series if x >= 0
                            ))) - 1),
                        ),
                    )
                input_values[bedrooms_col] = chosen_bedrooms

            if bathrooms_col:
                bath_series = pd.to_numeric(
                    valid_data[bathrooms_col], errors="coerce"
                ).dropna()
                bath_options = sorted(set(float(x) for x in bath_series if x >= 0))
                with col_b:
                    chosen_baths = st.selectbox(
                        "Full bathrooms",
                        bath_options,
                        index=min(1, len(bath_options) - 1) if bath_options else 0,
                    )
                input_values[bathrooms_col] = chosen_baths

            if garage_col:
                garage_series = pd.to_numeric(
                    valid_data[garage_col], errors="coerce"
                ).dropna()
                garage_options = sorted(set(float(x) for x in garage_series if x >= 0))
                with col_a:
                    chosen_garage = st.selectbox(
                        "Garage capacity",
                        garage_options,
                        index=min(1, len(garage_options) - 1) if garage_options else 0,
                    )
                input_values[garage_col] = chosen_garage

            if year_col:
                years = pd.to_numeric(valid_data[year_col], errors="coerce").dropna()
                with col_b:
                    chosen_year = st.number_input(
                        "Year built",
                        min_value=int(years.min()),
                        max_value=2026,
                        value=int(years.median()),
                        step=1,
                    )
                input_values[year_col] = chosen_year

            if neighborhood_col:
                neighborhoods = sorted(
                    valid_data[neighborhood_col].dropna().astype(str).unique().tolist()
                )
                default_neighborhood = str(base_values[neighborhood_col])
                default_index = (
                    neighborhoods.index(default_neighborhood)
                    if default_neighborhood in neighborhoods else 0
                )
                chosen_neighborhood = st.selectbox(
                    "Neighborhood",
                    neighborhoods,
                    index=default_index,
                )
                input_values[neighborhood_col] = chosen_neighborhood

            if basement_col:
                basement_values = pd.to_numeric(
                    valid_data[basement_col], errors="coerce"
                ).dropna()
                with col_a:
                    chosen_basement = st.number_input(
                        "Basement area (sq ft)",
                        min_value=0,
                        max_value=10000,
                        value=int(basement_values.median()),
                        step=50,
                    )
                input_values[basement_col] = chosen_basement

            if lot_col:
                lot_values = pd.to_numeric(valid_data[lot_col], errors="coerce").dropna()
                with col_b:
                    chosen_lot = st.number_input(
                        "Lot area (sq ft)",
                        min_value=500,
                        max_value=100000,
                        value=int(lot_values.median()),
                        step=100,
                    )
                input_values[lot_col] = chosen_lot

            submitted = st.form_submit_button(
                "Estimate property value  →",
                type="primary",
                use_container_width=True,
            )

    with right:
        st.markdown(
            '<div class="panel-title">Your valuation</div>'
            '<div class="panel-sub">Prediction generated by your saved ML pipeline.</div>',
            unsafe_allow_html=True,
        )

        if submitted:
            input_df = pd.DataFrame(
                [[input_values[col] for col in feature_columns]],
                columns=feature_columns,
            )

            try:
                prediction = float(model.predict(input_df)[0])

                if prediction <= 0:
                    st.warning("The model returned a non-positive price estimate. Review the inputs and model.")
                else:
                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_property"] = input_values.copy()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        prediction = st.session_state.get("last_prediction")

        if prediction is not None:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">Estimated market value</div>
                    <div class="price-value">{money(prediction)}</div>
                    <div class="price-note">
                        Model-generated estimate based on the selected property features.
                        This is not a formal appraisal or guaranteed selling price.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            delta = prediction - median_price
            direction = "above" if delta >= 0 else "below"
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">Compared with dataset median</div>
                    <div style="font-size:1.5rem;font-weight:800;color:#45e0a0">
                        {money(abs(delta))} {direction}
                    </div>
                    <div class="panel-sub">
                        Reference median: {money(median_price)}.
                        This comparison describes the dataset, not the entire real-estate market.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Estimate context**")
            st.write(f"- Model output: {money(prediction)}")
            st.write(f"- Dataset median: {money(median_price)}")
            st.write(f"- Features supplied: {len(feature_columns)}")
        else:
            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-label">Estimated market value</div>
                    <div class="price-value">—</div>
                    <div class="price-note">
                        Configure the property and select Estimate property value
                        to generate a prediction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-title">Dataset reference</div>
                    <div style="font-size:1.65rem;font-weight:800;color:#edf3ff">
                        {money(median_price)}
                    </div>
                    <div class="panel-sub">Observed median sale price in the dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "The estimator uses the saved model pipeline. It does not fetch live listings "
            "or current market prices."
        )

# --------------------------------------------------
# MARKET INSIGHTS
# --------------------------------------------------

elif page == "Market Insights":
    st.markdown('<div class="eyebrow">DATA EXPLORER / 03</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Understand the market.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Explore patterns in the recorded Ames housing sales.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("MEDIAN PRICE", money(median_price), "Dataset observation")
    with c2:
        metric_card("LOWEST RECORDED", money(valid_data["SalePrice"].min()), "Dataset minimum")
    with c3:
        metric_card("HIGHEST RECORDED", money(valid_data["SalePrice"].max()), "Dataset maximum")

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="panel-title">Living area vs sale price</div>'
            '<div class="panel-sub">Explore how above-ground living space relates to price.</div>',
            unsafe_allow_html=True,
        )

        if area_col:
            plot_data = valid_data[[area_col, "SalePrice"]].copy()
            plot_data[area_col] = pd.to_numeric(plot_data[area_col], errors="coerce")
            plot_data = plot_data.dropna()

            fig = px.scatter(
                plot_data,
                x=area_col,
                y="SalePrice",
                opacity=0.55,
                labels={
                    area_col: "Living area (sq ft)",
                    "SalePrice": "Sale price (USD)",
                },
                color_discrete_sequence=["#45e0a0"],
            )
            st.plotly_chart(make_chart_theme(fig), use_container_width=True)
        else:
            st.info("Living area column not found.")

    with right:
        st.markdown(
            '<div class="panel-title">Neighborhood price comparison</div>'
            '<div class="panel-sub">Median recorded price for each neighborhood.</div>',
            unsafe_allow_html=True,
        )

        if neighborhood_col:
            neighborhood_stats = (
                valid_data.groupby(neighborhood_col, dropna=True)["SalePrice"]
                .median()
                .sort_values(ascending=True)
                .reset_index()
            )
            neighborhood_stats.columns = ["Neighborhood", "Median Price"]
            fig = px.bar(
                neighborhood_stats,
                x="Median Price",
                y="Neighborhood",
                orientation="h",
                labels={
                    "Median Price": "Median sale price (USD)",
                    "Neighborhood": "",
                },
                color_discrete_sequence=["#45e0a0"],
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(make_chart_theme(fig, height=480), use_container_width=True)
        else:
            st.info("Neighborhood column not found.")

    st.markdown("### Dataset preview")
    st.dataframe(
        valid_data.head(15),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = valid_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned dataset as CSV",
        data=csv_bytes,
        file_name="ames_housing_data.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":
    st.markdown('<div class="eyebrow">MODEL LAB / 04</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">Inside the prediction engine.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Evaluation metrics and model comparison from your training run.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if comparison is not None and not comparison.empty:
        st.markdown("### Evaluation results")
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        numeric_cols = comparison.select_dtypes(include="number").columns.tolist()
        name_col = next(
            (c for c in comparison.columns if c.lower() in ["model", "model_name", "algorithm"]),
            None,
        )

        if name_col and numeric_cols:
            metric_options = [
                c for c in numeric_cols
                if any(token in c.lower() for token in ["mae", "rmse", "r2", "score"])
            ]

            if metric_options:
                selected_metric = st.selectbox(
                    "Compare models by metric",
                    metric_options,
                )

                fig = px.bar(
                    comparison,
                    x=name_col,
                    y=selected_metric,
                    color_discrete_sequence=["#45e0a0"],
                    labels={name_col: "Model", selected_metric: selected_metric},
                    text_auto=".4f",
                )
                st.plotly_chart(make_chart_theme(fig), use_container_width=True)

        st.caption(
            "Use the exact metric names and test-set results produced by your training script. "
            "Lower MAE/RMSE is better; higher R² is generally better."
        )
    else:
        st.warning(
            "No model comparison file was found. Train the models first to generate "
            "results/model_comparison.csv."
        )
        st.code("python .\\src\\train_model.py", language="powershell")

    st.markdown("### How the system works")

    steps = [
        ("01", "Input", "Property features are collected through the estimator form."),
        ("02", "Preprocessing", "The saved pipeline handles missing values and categorical features."),
        ("03", "Prediction", "The trained regression model estimates the sale price."),
        ("04", "Evaluation", "Test-set metrics summarize predictive performance."),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="panel" style="padding:1rem 1.2rem">
                <div style="display:flex;gap:1rem;align-items:flex-start">
                    <div style="color:#45e0a0;font-weight:800;font-size:1.1rem">{number}</div>
                    <div>
                        <div class="panel-title">{title}</div>
                        <div class="panel-sub" style="margin-bottom:0">{description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#71839d;font-size:0.78rem;padding:0.5rem 0 1rem">
        ESTATIQ · Machine Learning Property Valuation · Educational project using Ames housing data
    </div>
    """,
    unsafe_allow_html=True,
)
