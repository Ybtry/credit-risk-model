import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_column='TransactionStartTime'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column], errors='coerce', utc=True)
        X_copy['transaction_hour'] = X_copy[self.time_column].dt.hour
        X_copy['transaction_day_of_week'] = X_copy[self.time_column].dt.dayofweek
        X_copy['transaction_month'] = X_copy[self.time_column].dt.month
        X_copy['transaction_year'] = X_copy[self.time_column].dt.year
        
        # --- NEW: Drop the original TransactionStartTime column ---
        X_copy = X_copy.drop(columns=[self.time_column], errors='ignore')
        
        return X_copy

class AmountHandler(BaseEstimator, TransformerMixin):
    def __init__(self, amount_column='Amount', value_column='Value'):
        self.amount_column = amount_column
        self.value_column = value_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['is_refund'] = (X_copy[self.amount_column] < 0).astype(int)
        return X_copy

def get_data_processing_pipeline():
    numerical_features = ['Amount', 'Value', 'PricingStrategy']
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId', 'ProductId']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('datetime_extractor', DateTimeExtractor()),
        ('amount_handler', AmountHandler()),
        ('preprocessor', preprocessor)
    ])

    return pipeline

if __name__ == "__main__":
    print("Running data_processing.py directly for testing...")

    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5'],
        'AccountId': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'SubscriptionId': ['S1', 'S2', 'S3', 'S4', 'S5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256, 256, 256],
        'ProviderId': ['P1', 'P2', 'P1', 'P3', 'P2'],
        'ProductId': ['ProdA', 'ProdB', 'ProdA', 'ProdC', 'ProdB'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility_bill', 'financial_services'],
        'ChannelId': ['ChannelId_3', 'ChannelId_2', 'ChannelId_3', 'ChannelId_1', 'ChannelId_2'],
        'Amount': [1000.0, -50.0, 500.0, 20000.0, -100.0],
        'Value': [1000, 50, 500, 20000, 100],
        'TransactionStartTime': ['2023-01-01T10:00:00Z', '2023-01-01T11:00:00Z', '2023-01-02T12:00:00Z', '2023-01-02T13:00:00Z', '2023-01-03T14:00:00Z'],
        'PricingStrategy': [2, 2, 0, 1, 2],
        'FraudResult': [0, 0, 0, 1, 0]
    }
    dummy_df = pd.DataFrame(data)

    processor_pipeline = get_data_processing_pipeline()
    processed_data_array = processor_pipeline.fit_transform(dummy_df)

    print("\nShape of processed data (array):", processed_data_array.shape)
    print("\nFirst 5 rows of processed data (array):")
    print(processed_data_array[:5])

    try:
        feature_names_out = processor_pipeline.named_steps['preprocessor'].get_feature_names_out()
        print("\nFeature names out (partial):", feature_names_out)
    except AttributeError:
        print("\nCould not retrieve all feature names easily from the transformed array.")

    print("\nData processing pipeline test complete.")
