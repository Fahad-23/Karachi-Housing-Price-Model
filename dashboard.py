"""Streamlit dashboard for Karachi Housing Price Prediction."""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import RAW_DATA_PATH, ARTIFACTS_PATH
from src.model import load_all_models, predict
from src.preprocessing import load_and_clean

st.set_page_config(
    page_title="Karachi Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def get_models():
    if not ARTIFACTS_PATH.exists():
        return None, None
    return load_all_models()


@st.cache_data
def get_data():
    if not RAW_DATA_PATH.exists():
        return None
    return load_and_clean(str(RAW_DATA_PATH))


pipelines, metadata = get_models()
df = get_data()

if pipelines is None or df is None:
    st.error("Model or data not found. Run `python train.py` first.")
    st.stop()

best_model = metadata["best_model"]

# --- Header ---
st.title("Karachi Housing Price Predictor")
st.markdown(f"**Best model:** {best_model} | "
            f"**R2:** {metadata['all_results'][best_model]['r2']} | "
            f"**Trained on:** {metadata['train_size']:,} samples")

st.divider()

# --- Layout: Prediction + Model Info ---
col_predict, col_info = st.columns([3, 2])

with col_predict:
    st.subheader("Predict Property Price")

    # Model selection — best is default
    model_options = metadata["available_models"]
    default_idx = model_options.index(best_model)
    labels = [f"{m} (Recommended)" if m == best_model else m for m in model_options]
    selected_label = st.selectbox("Model", labels, index=default_idx)
    selected_model = model_options[labels.index(selected_label)]

    prop_type = st.selectbox("Property Type", metadata["known_types"])

    location = st.selectbox(
        "Location",
        metadata["known_locations"],
        help="Select the area/locality in Karachi",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        area = st.number_input("Area (sq. yards)", min_value=10, max_value=10000, value=250, step=10)
    with col_b:
        beds = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    with col_c:
        baths = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)

    if st.button("Predict Price", type="primary", use_container_width=True):
        predicted = predict(pipelines[selected_model], prop_type, location, area, beds, baths)
        predicted = max(0, predicted)

        if predicted >= 1e7:
            formatted = f"{predicted / 1e7:.2f} Crore"
        elif predicted >= 1e5:
            formatted = f"{predicted / 1e5:.2f} Lakh"
        else:
            formatted = f"{predicted:,.0f}"

        st.success(f"**Predicted Price: PKR {formatted}**")
        st.caption(f"Raw: PKR {predicted:,.0f} | Model: {selected_model}")

        if selected_model != best_model:
            st.info(f"Note: You are using {selected_model}. The recommended model is {best_model}.")

with col_info:
    st.subheader("Model Performance")

    results_df = pd.DataFrame(metadata["all_results"]).T
    results_df.index.name = "Model"

    st.dataframe(
        results_df[["r2", "rmse", "mae"]].style
        .highlight_max(subset=["r2"], color="#90EE90")
        .highlight_min(subset=["rmse", "mae"], color="#90EE90")
        .format({"r2": "{:.4f}", "rmse": "{:,.0f}", "mae": "{:,.0f}"}),
        use_container_width=True,
    )

    st.caption(f"Best model selected by highest R2 on test set ({metadata['test_size']:,} samples)")

st.divider()

# --- Data Exploration ---
st.subheader("Dataset Exploration")

tab_dist, tab_location, tab_scatter = st.tabs(["Price Distribution", "By Location", "Area vs Price"])

with tab_dist:
    fig = px.histogram(
        df, x="Price", nbins=50,
        title="Property Price Distribution",
        labels={"Price": "Price (PKR)"},
        color="Type",
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with tab_location:
    top_locations = df["Location_Clean"].value_counts().head(20)
    median_prices = df[df["Location_Clean"].isin(top_locations.index)].groupby("Location_Clean")["Price"].median()
    median_prices = median_prices.sort_values(ascending=True)

    fig = px.bar(
        x=median_prices.values,
        y=median_prices.index,
        orientation="h",
        title="Median Price by Location (Top 20 by listing count)",
        labels={"x": "Median Price (PKR)", "y": "Location"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_scatter:
    fig = px.scatter(
        df, x="Area", y="Price",
        color="Type",
        title="Area vs Price",
        labels={"Area": "Area (sq. yards)", "Price": "Price (PKR)"},
        opacity=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.divider()
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption(f"Dataset: {len(df):,} properties | {len(df['Location_Clean'].unique())} locations")
with col_f2:
    st.caption(f"Model trained: {metadata['trained_at'][:10]}")
