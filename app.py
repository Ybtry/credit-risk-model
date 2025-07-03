# app.py
import pandas as pd
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

project_root_dir = os.getcwd()
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from src.data_processing import get_data_processing_pipeline

app = Flask(__name__)

loaded_model = None
data_processing_pipeline = None
feature_names_in_order = None

def load_model_and_pipeline():
    global loaded_model, data_processing_pipeline, feature_names_in_order

    print("Loading model and data processing pipeline...")

    mlflow_tracking_uri = os.path.join(project_root_dir, 'notebooks', 'mlruns')
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

    client = mlflow.tracking.MlflowClient()
    experiment_name = "Credit_Risk_Model_Training"

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError(f"No runs found for experiment '{experiment_name}'. Please train a model first.")

        latest_run = runs[0]
        latest_run_id = latest_run.info.run_id
        print(f"Latest MLflow Run ID: {latest_run_id}")

        model_uri = f"runs:/{latest_run_id}/logistic_regression_model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")

        data_processing_pipeline = get_data_processing_pipeline()
        print("Data processing pipeline initialized.")

        raw_data_path = os.path.join(project_root_dir, 'data', 'raw', 'data.csv')
        try:
            full_raw_df_for_fitting = pd.read_csv(raw_data_path)
            
            cols_to_drop_for_fitting = [
                'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                'CurrencyCode', 'CountryCode', 'FraudResult'
            ]
            full_raw_df_for_fitting = full_raw_df_for_fitting.drop(columns=[col for col in cols_to_drop_for_fitting if col in full_raw_df_for_fitting.columns], errors='ignore')

            data_processing_pipeline.fit(full_raw_df_for_fitting.copy())
            print("Data processing pipeline fitted on FULL raw data (with non-feature columns dropped).")
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {raw_data_path}. Cannot fit pipeline.")
            loaded_model = None
            data_processing_pipeline = None
            return
        except Exception as e:
            print(f"Error fitting pipeline on full data: {e}")
            loaded_model = None
            data_processing_pipeline = None
            return

        if hasattr(loaded_model, 'feature_names_in_'):
            feature_names_in_order = loaded_model.feature_names_in_
            print("Feature names inferred from loaded model.")
        else:
            print("Warning: Model does not have 'feature_names_in_'. Ensure input features match training.")

    except Exception as e:
        print(f"Error during model or pipeline loading: {e}")
        loaded_model = None
        data_processing_pipeline = None
        feature_names_in_order = None


@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or data_processing_pipeline is None:
        return jsonify({"error": "Model or pipeline not loaded. Please check server logs."}), 500

    try:
        json_data = request.get_json(force=True)
        if not isinstance(json_data, list):
            json_data = [json_data]

        input_df = pd.DataFrame(json_data)

        cols_to_drop = [
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
            'CurrencyCode', 'CountryCode', 'FraudResult'
        ]
        
        input_df_for_processing = input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], errors='ignore')

        processed_input_array = data_processing_pipeline.transform(input_df_for_processing.copy())

        predictions = loaded_model.predict(processed_input_array)
        probabilities = loaded_model.predict_proba(processed_input_array)[:, 1]

        results = []
        for i in range(len(predictions)):
            results.append({
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i])
            })

        return jsonify(results)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    load_model_and_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=True)
