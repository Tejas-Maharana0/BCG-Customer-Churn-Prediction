from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import os


BASE_DIR = Path(__file__).resolve().parent
API_URL = (st.secrets.get("API_URL") or os.getenv("API_URL") or "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Customer Churn Prediction | BCG",
    page_icon="📊",
    layout="wide",
)

st.title("�� Customer Churn Prediction")
st.caption("Machine Learning • Customer Analytics • Churn Risk Assessment")

st.info(
    "Upload a CSV containing the engineered features expected by the trained model. "
    "The app adds predictions to your uploaded rows; it does not modify the source file."
)

with st.sidebar:
    st.header("API Connection")
    st.write(f"Backend: `{API_URL}`")
    if st.button("Check API Status", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if health.get("model_available"):
                st.success("API connected and model files are available.")
            else:
                st.warning("API is running, but model files are missing.")
        except requests.RequestException as exc:
            st.error(f"Cannot connect to API: {exc}")

uploaded_file = st.file_uploader(
    "Upload engineered customer data (CSV)",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()

    st.subheader("Data Preview")
    st.write(f"Rows: **{len(data):,}** | Columns: **{len(data.columns):,}**")
    st.dataframe(data.head(10), use_container_width=True)

    if st.button("Predict Churn", type="primary", use_container_width=True):
        if data.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        try:
            response = requests.get(f"{API_URL}/features", timeout=10)
            response.raise_for_status()
            required_features = response.json()["features"]

            missing = [col for col in required_features if col not in data.columns]
            if missing:
                st.error(
                    "This CSV is missing required model features. "
                    "Upload the engineered dataset used for training."
                )
                with st.expander("View missing feature names"):
                    st.write(missing)
                st.stop()

            # Process the CSV in smaller batches to avoid large-request failures.
            batch_size = 500
            results = []
            total_rows = len(data)
            progress_bar = st.progress(0, text="Preparing predictions...")
            status_text = st.empty()

            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = data.iloc[start:end][required_features]
                batch = batch.astype(object).where(pd.notna(batch), None)
                payload = {"records": batch.to_dict(orient="records")}

                status_text.info(
                    f"Predicting rows {start + 1:,}–{end:,} of {total_rows:,}..."
                )

                prediction_response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=90,
                )
                prediction_response.raise_for_status()
                batch_results = prediction_response.json()["predictions"]

                if len(batch_results) != len(batch):
                    raise ValueError(
                        f"Backend returned {len(batch_results)} predictions "
                        f"for {len(batch)} input rows."
                    )

                results.extend(batch_results)
                progress_bar.progress(
                    min(end / total_rows, 1.0),
                    text=f"Processed {end:,} of {total_rows:,} rows",
                )

            status_text.success(
                f"Successfully generated predictions for {len(results):,} rows."
            )

            result_df = data.copy()
            result_df["Predicted_Churn"] = [
                result["churn_label"] for result in results
            ]
            result_df["Churn_Probability"] = [
                result.get("churn_probability") for result in results
            ]

            st.subheader("Prediction Results")
            total = len(result_df)
            churn_count = int((result_df["Predicted_Churn"] == "Yes").sum())
            no_churn_count = total - churn_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Customers Analyzed", f"{total:,}")
            col2.metric("Predicted Churn", f"{churn_count:,}")
            col3.metric("Predicted No Churn", f"{no_churn_count:,}")

            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_data,
                file_name="customer_churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except requests.RequestException as exc:
            st.error(
                "Could not get predictions from the backend. "
                "Make sure the FastAPI server is running. "
                f"Details: {exc}"
            )
        except (KeyError, ValueError) as exc:
            st.error(f"Unexpected API response: {exc}")
