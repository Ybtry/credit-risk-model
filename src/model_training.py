import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import mlflow
import mlflow.sklearn
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_processing import get_data_processing_pipeline, generate_proxy_target

def train_and_log_model(data_path, model_name="LogisticRegressionModel", test_size=0.2, random_state=42, C=1.0):
    mlflow.set_experiment("Credit_Risk_Model_Training")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("C", C)

        print(f"Loading data from: {data_path}")
        try:
            raw_df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Please ensure it exists.")
            return

        pipeline = get_data_processing_pipeline()
        processed_data_array = pipeline.fit_transform(raw_df.copy())

        preprocessor_step = pipeline.named_steps['preprocessor']
        num_features_out = preprocessor_step.named_transformers_['num'].get_feature_names_out()
        cat_features_out = preprocessor_step.named_transformers_['cat'].get_feature_names_out()

        df_before_preprocessor = pipeline.named_steps['datetime_extractor'].transform(raw_df.copy())
        df_before_preprocessor = pipeline.named_steps['amount_handler'].transform(df_before_preprocessor)
        processed_by_preprocessor = ['Amount', 'Value', 'PricingStrategy',
                                     'ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']
        passthrough_cols_final = [col for col in df_before_preprocessor.columns if col not in processed_by_preprocessor]
        final_feature_names = list(num_features_out) + list(cat_features_out) + passthrough_cols_final

        processed_df = pd.DataFrame(processed_data_array, columns=final_feature_names)

        numeric_cols_to_convert = list(num_features_out) + \
                                  list(cat_features_out) + \
                                  ['transaction_hour', 'transaction_day_of_week', 'transaction_month',
                                   'transaction_year', 'is_refund', 'FraudResult']

        for col in numeric_cols_to_convert:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        proxy_target_df = generate_proxy_target(raw_df.copy())
        proxy_target_df_reset = proxy_target_df.reset_index()

        final_processed_df = pd.merge(
            processed_df,
            proxy_target_df_reset[['CustomerId', 'is_high_risk']],
            on='CustomerId',
            how='left'
        )

        X = final_processed_df.drop(columns=[
            'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
            'CurrencyCode', 'CountryCode', 'TransactionStartTime', 'FraudResult',
            'is_high_risk'
        ])
        y = final_processed_df['is_high_risk']

        initial_rows = X.shape[0]
        X = X.dropna(axis=1)
        X = X.dropna(axis=0)
        rows_after_na = X.shape[0]
        if initial_rows != rows_after_na:
            print(f"Dropped {initial_rows - rows_after_na} rows due to NaN values after processing.")

        y = y.loc[X.index]

        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Target distribution (is_high_risk):\n{y.value_counts(normalize=True)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

        model = LogisticRegression(random_state=random_state, C=C, solver='liblinear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"PR AUC Score: {pr_auc:.4f}")

        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "precision_0": report['0']['precision'],
            "recall_0": report['0']['recall'],
            "f1-score_0": report['0']['f1-score'],
            "precision_1": report['1']['precision'],
            "recall_1": report['1']['recall'],
            "f1-score_1": report['1']['f1-score'],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print("Model training and logging complete.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_file_path = os.path.join(current_dir, '..', 'data', 'raw', 'data.csv')

    train_and_log_model(raw_data_file_path)
