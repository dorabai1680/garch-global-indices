from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@st.cache_data
def load_outputs(reports_dir: Path):
    required = {
        "summary": reports_dir / "summary.csv",
        "returns": reports_dir / "returns_and_volatility.csv",
        "evaluation": reports_dir / "forecast_evaluation.csv",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing generated outputs: {', '.join(missing)}. Run `python run_analysis.py --demo` first."
        )
    metadata_path = reports_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return (
        pd.read_csv(required["summary"]),
        pd.read_csv(required["returns"], parse_dates=["date"]),
        pd.read_csv(required["evaluation"]),
        metadata,
    )


def main() -> None:
    st.set_page_config(page_title="Global GARCH Monitor", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #f7fafc 0%, #ffffff 28%); }
        [data-testid="stMetric"] { background: white; border: 1px solid #e5eaf0;
            padding: 1rem; border-radius: .75rem; box-shadow: 0 4px 18px rgba(20,35,50,.05); }
        [data-testid="stSidebar"] { border-right: 1px solid #e5eaf0; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Global GARCH Volatility Monitor")
    st.caption("Compare conditional volatility, persistence, and out-of-sample forecast accuracy.")

    try:
        summary, observations, evaluation, metadata = load_outputs(REPORTS_DIR)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    index_names = summary["index"].tolist()
    source = metadata.get("data_source", "generated analysis outputs")
    st.sidebar.caption(f"Data source: {source}")
    selected = st.sidebar.multiselect("Indices", index_names, default=index_names)
    if not selected:
        st.info("Select at least one index from the sidebar.")
        st.stop()

    min_date, max_date = observations["date"].min().date(), observations["date"].max().date()
    selected_dates = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(selected_dates) != 2:
        st.info("Choose both a start and end date.")
        st.stop()
    start_date, end_date = selected_dates
    filtered_summary = summary[summary["index"].isin(selected)]
    filtered_observations = observations[
        observations["index"].isin(selected)
        & observations["date"].dt.date.between(start_date, end_date)
    ]
    filtered_evaluation = evaluation[evaluation["index"].isin(selected)]
    if filtered_observations.empty:
        st.warning("No observations are available in this date range.")
        st.stop()

    most_persistent = filtered_summary.loc[filtered_summary["persistence_alpha_plus_beta"].idxmax()]
    latest = filtered_observations.sort_values("date").groupby("index", as_index=False).tail(1)
    latest_mean = latest["conditional_volatility"].mean()
    best_qlike_row = filtered_evaluation.loc[filtered_evaluation["QLIKE"].idxmin()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Most persistent", most_persistent["index"], f"α + β = {most_persistent['persistence_alpha_plus_beta']:.3f}")
    col2.metric("Latest mean volatility", f"{latest_mean:.2f}%")
    col3.metric("Lowest QLIKE", best_qlike_row["method"], best_qlike_row["index"])

    st.subheader("Conditional volatility")
    volatility_chart = filtered_observations.pivot(index="date", columns="index", values="conditional_volatility")
    st.line_chart(volatility_chart, height=410)

    left, right = st.columns(2)
    with left:
        st.subheader("Volatility persistence")
        persistence = filtered_summary.set_index("index")[["persistence_alpha_plus_beta"]]
        st.bar_chart(persistence, height=330)
    with right:
        st.subheader("Forecast comparison")
        metric = st.selectbox("Loss metric", ["QLIKE", "RMSE", "MAE"])
        comparison = filtered_evaluation.pivot(index="index", columns="method", values=metric)
        st.bar_chart(comparison, height=290)

    with st.expander("Model parameters and forecast scores"):
        st.markdown("**GARCH parameter estimates**")
        st.dataframe(filtered_summary, hide_index=True, width="stretch")
        st.markdown("**Out-of-sample evaluation**")
        styled = filtered_evaluation.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "QLIKE": "{:.4f}"})
        st.dataframe(styled, hide_index=True, width="stretch")

    st.caption(
        f"Observations: {filtered_observations['date'].min():%Y-%m-%d} to "
        f"{filtered_observations['date'].max():%Y-%m-%d} · Source: {source}. "
        "Lower forecast-loss values are better."
    )


if __name__ == "__main__":
    main()
