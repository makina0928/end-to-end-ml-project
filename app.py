import streamlit as st
import pandas as pd
import os
import sys

from machinelearning.exception.exception import MachinelearningException
from machinelearning.logging.logger import logging
from machinelearning.pipeline.training_pipeline import TrainingPipeline
from machinelearning.utils.main_utils.utils import load_object, load_numpy_array_data

from machinelearning.components.model_trainer import NetworkModel


st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction App")
st.write("Streamlit interface for model training and prediction.")

# ---------------------------------------------------------
# TRAINING SECTION
# ---------------------------------------------------------
st.header("Train Model")

if st.button("Start Training"):
    try:
        with st.spinner("Training in progress..."):
            pipeline = TrainingPipeline()
            trainer_artifact = pipeline.run_pipeline()

        st.success("Training Completed Successfully!")

        # -----------------------------------------------------
        # HORIZONTAL METRIC SECTIONS WITH COLORED CARDS
        # -----------------------------------------------------
        col1, col2, col3 = st.columns(3)

        # ---------- BEST MODEL CARD ----------
        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color:#e3f2fd;
                    padding:20px;
                    border-radius:15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                    <h4 style="margin:0; color:#0d47a1;">Best Model</h4>
                    <p style="font-size:18px; margin-top:10px;">
                        <b>{trainer_artifact.best_model_name}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- TRAIN METRICS CARD ----------
        with col2:
            train_f1 = round(trainer_artifact.train_metric_artifact.f1_score, 3)
            train_precision = round(trainer_artifact.train_metric_artifact.precision_score, 3)
            train_recall = round(trainer_artifact.train_metric_artifact.recall_score, 3)

            st.markdown(
                f"""
                <div style="
                    background-color:#e8f5e9;
                    padding:20px;
                    border-radius:15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                    <h4 style="margin:0; color:#1b5e20;">Train Metrics</h4>
                    <p><b>F1 Score:</b> {train_f1}</p>
                    <p><b>Precision:</b> {train_precision}</p>
                    <p><b>Recall:</b> {train_recall}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------- TEST METRICS CARD ----------
        with col3:
            test_f1 = round(trainer_artifact.test_metric_artifact.f1_score, 3)
            test_precision = round(trainer_artifact.test_metric_artifact.precision_score, 3)
            test_recall = round(trainer_artifact.test_metric_artifact.recall_score, 3)

            st.markdown(
                f"""
                <div style="
                    background-color:#f3e5f5;
                    padding:20px;
                    border-radius:15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                    <h4 style="margin:0; color:#4a148c;">Test Metrics</h4>
                    <p><b>F1 Score:</b> {test_f1}</p>
                    <p><b>Precision:</b> {test_precision}</p>
                    <p><b>Recall:</b> {test_recall}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------
        # CONFUSION MATRIX
        # -----------------------------------------------------
        st.subheader("Confusion Matrix on Test Set")

        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load test predictions
        y_test_pred = load_object(trainer_artifact.y_test_pred_path)

        # Load true labels
        test_arr = load_numpy_array_data(trainer_artifact.test_array_path)
        y_test_true = test_arr[:, -1]

        cm = confusion_matrix(y_test_true, y_test_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Training Failed: {e}")
        logging.error(e)


# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------
st.header("Predict Using CSV File")

uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # ---------------------------------------------------
        # Load preprocessor + model separately and combine into NetworkModel
        # ---------------------------------------------------
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        st.success("Model Loaded Successfully!")

        if st.button("Run Prediction"):
            with st.spinner("Predicting..."):
                predictions = network_model.predict(df)

                df["predicted_column"] = predictions

                # Ensure output directory exists
                os.makedirs("prediction_output", exist_ok=True)

                output_path = "prediction_output/predictions.csv"
                df.to_csv(output_path, index=False)

                st.success("Prediction Complete!")
                st.subheader("Prediction Results")
                st.dataframe(df)

                st.download_button(
                    label="Download Prediction CSV",
                    data=df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logging.error(e)
