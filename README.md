Credit Risk Probability Model for Alternative Data
Project Overview
This project focuses on building an end-to-end credit risk probability model for Bati Bank, a leading financial service provider. The goal is to develop a buy-now-pay-later service by creating a Credit Scoring Model that leverages alternative data provided by an upcoming successful eCommerce company.

Traditionally, credit scoring models rely on past borrower behavior. Our key innovation lies in transforming raw eCommerce transaction data into a predictive risk signal. By analyzing customer Recency, Frequency, and Monetary (RFM) patterns, we'll engineer a proxy for credit risk. This allows us to train a model that outputs a risk probability score, a vital metric that will inform loan approvals and terms for new customers.

Key Objectives Achieved:
Define a proxy variable for categorizing users as high-risk ("bad") or low-risk ("good") based on RFM analysis.

Select observable features from raw transaction data and engineer new ones (e.g., time-based, refund status) that are strong predictors of the defined risk variable.

Develop, train, and evaluate a robust Logistic Regression model to assign a risk probability for new customers, specifically addressing class imbalance.

Interpret the model's key features to understand drivers of credit risk.

Deploy the trained model as a scalable API endpoint for real-time risk predictions.

Outline strategies for Continuous Integration/Continuous Deployment (CI/CD) and Model Monitoring to ensure long-term model performance and reliability in production.

Project Structure
The project follows a standardized structure to ensure maintainability, modularity, and scalability:

credit-risk-model/
├── data/                         # Data storage (ignored by Git)
│   ├── raw/                      # Raw, untransformed source data (e.g., data.csv)
│   └── processed/                # Cleaned and processed data (not explicitly saved, handled by pipeline)
├── notebooks/
│   ├── 1.0-eda.ipynb             # Jupyter notebook for exploratory data analysis
│   └── 4.0-model-evaluation-and-interpretation.ipynb # Notebook for data processing, model training, evaluation, and interpretation
├── src/                          # Source code for the application's core logic
│   ├── __init__.py               # Python package initialization file
│   ├── data_processing.py        # Script for data loading, cleaning, feature engineering, and target variable creation
│   └── model_training.py         # Script for model training, evaluation, and MLflow tracking
├── app.py                        # Flask application for exposing the model via an API endpoint
├── requirements.txt              # Lists all Python package dependencies for the project
├── .gitignore                    # Specifies files and directories that Git should ignore
├── test_data.json                # Sample JSON payload for testing the prediction API
└── README.md                     # Project overview, setup guide, and key documentation

Prerequisites
Before you begin, ensure you have the following installed:

Python 3.9+: The primary programming language for this project.

Git: For version control and managing the repository.

pip: Python package installer.

Setup Instructions
Follow these steps to set up the project locally:

Clone the repository:

git clone https://github.com/Ybtry/credit-risk-model.git
cd credit-risk-model

Create and activate a Python virtual environment:
It's crucial to use a virtual environment to manage project dependencies isolation.

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
Once your virtual environment is active, install all required Python packages:

pip install -r requirements.txt

(You may need to create a requirements.txt file if you haven't already. You can generate one using pip freeze > requirements.txt after installing all necessary libraries like pandas, numpy, scikit-learn, mlflow, matplotlib, seaborn, Flask.)

Data Acquisition
The raw transaction data (data.csv) required for this challenge can be obtained from the Kaggle platform:

Download from Kaggle: Xente Challenge | Kaggle

Placement: After downloading, rename the file to data.csv (if necessary) and place it into the data/raw/ directory within your project structure.

Project Walkthrough
This section guides you through the key phases of the project.

1. Exploratory Data Analysis (EDA)
Notebook: notebooks/1.0-eda.ipynb

Description: This notebook performs initial data loading, cleaning, and statistical analysis to understand the dataset's characteristics, identify potential issues (missing values, outliers), and gain insights into transaction patterns.

2. Feature Engineering & Proxy Target Variable
Script: src/data_processing.py

Description: This script contains the core logic for transforming raw data into features suitable for machine learning. It includes:

DateTimeExtractor: Extracts temporal features (hour, day of week, month, year) from transaction timestamps.

AmountHandler: Derives features like is_refund from transaction amounts.

ColumnTransformer: Applies scaling to numerical features and one-hot encoding to categorical features.

Proxy Target: A proxy variable is_high_risk is engineered based on RFM (Recency, Frequency, Monetary) analysis to serve as the target for the credit risk model, as direct default labels are unavailable.

3. Model Training, Evaluation, and Interpretation
Notebook: notebooks/4.0-model-evaluation-and-interpretation.ipynb

Script: src/model_training.py

Description: This phase involves building and assessing the credit risk model.

Model: A Logistic Regression model is used due to its interpretability and effectiveness as a baseline.

Class Imbalance Handling: The model is trained with class_weight='balanced' to address the inherent imbalance between high-risk and low-risk customers, ensuring the model can effectively identify the minority class.

MLflow Tracking: Model training runs, parameters (e.g., C, class_weight), and performance metrics (Precision, Recall, F1-score, ROC AUC, PR AUC) are tracked using MLflow for reproducibility and comparison.

Evaluation: Comprehensive evaluation is performed using classification reports, confusion matrices, ROC curves, and Precision-Recall curves.

Threshold Tuning: The classification threshold is analyzed to optimize the balance between precision and recall based on business needs (e.g., maximizing F1-score).

Model Interpretation: Coefficients of the Logistic Regression model are analyzed to understand the impact and direction of influence of each feature on the predicted credit risk.

4. Model Deployment
Script: app.py

Description: The trained model is exposed as a RESTful API endpoint using Flask, allowing for real-time credit risk predictions.

Functionality: The API loads the latest model and the fitted data processing pipeline from MLflow at startup. The /predict endpoint accepts new transaction data, preprocesses it using the same pipeline, and returns a binary prediction (high-risk/low-risk) along with the associated probability.

To run the API:

Ensure your virtual environment is active.

Navigate to the project root directory.

Run: python3 app.py

The API will be available at http://localhost:5000/predict.

Example curl request to test the API:

curl -X POST -H "Content-Type: application/json" \
     -d '{
         "TransactionId": "T_NEW_EXAMPLE",
         "BatchId": "B_NEW_EXAMPLE",
         "AccountId": "A_NEW_EXAMPLE",
         "SubscriptionId": "S_NEW_EXAMPLE",
         "CustomerId": "C_NEW_EXAMPLE",
         "CurrencyCode": "UGX",
         "CountryCode": 256,
         "ProviderId": "P1",
         "ProductId": "ProdA",
         "ProductCategory": "airtime",
         "ChannelId": "ChannelId_3",
         "Amount": 1500.0,
         "Value": 1500,
         "TransactionStartTime": "2024-07-03T15:30:00Z",
         "PricingStrategy": 2,
         "FraudResult": 0
     }' \
     http://localhost:5000/predict

Credit Scoring Business Understanding 
This section provides critical context on the financial industry's approach to credit risk and the strategic decisions driving our model's development.

How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord mandates a robust framework for financial institutions to assess and manage credit risk, particularly under its Pillar 1 (Minimum Capital Requirements) and Pillar 2 (Supervisory Review Process). This emphasis directly drives the need for an interpretable and well-documented model. Regulatory bodies require a clear understanding of how models arrive at their risk assessments to ensure compliance, prevent systemic risks, and validate capital adequacy. An interpretable model allows Bati Bank to explain its credit decisions to regulators, auditors, and even customers, fostering transparency and trust. Furthermore, it facilitates rigorous model validation, stress testing, and the identification of potential biases or flaws, all of which are critical for effective risk management and maintaining the bank's financial stability in a regulated environment. Without clear documentation and interpretability, the model would operate as a "black box," making it impossible to satisfy regulatory scrutiny or explain loan outcomes.

Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Creating a proxy variable for "default" is necessary because the provided eCommerce dataset lacks an explicit label indicating whether a customer has defaulted on a loan or service. To build a supervised machine learning model, a target variable (our "ground truth") is indispensable. By defining "high-risk" customers based on observed behavioral patterns like Recency, Frequency, and Monetary (RFM) values, we infer a proxy for credit risk.

However, relying on a proxy variable introduces significant business risks. The primary risk is misclassification. If our proxy does not perfectly align with actual credit default, we face:

False Positives (Type I Error): Classifying a truly creditworthy customer as high-risk. This leads to denying credit to good customers, resulting in lost revenue opportunities for Bati Bank and potentially damaging customer relationships.

False Negatives (Type II Error): Classifying a truly high-risk customer as low-risk. This results in approving loans that are likely to default, leading to direct financial losses for the bank due to unrecovered principal and interest.
Beyond direct losses, making predictions based on an imperfect proxy can lead to model drift (as the relationship between the proxy and true default may evolve) and unintended biases if the proxy definition inadvertently correlates with protected characteristics. Ultimately, the model's effectiveness in mitigating true credit risk is contingent on how well this engineered proxy genuinely reflects actual default behavior.

What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
In a regulated financial context, the choice between simple, interpretable models (e.g., Logistic Regression with Weight of Evidence - WoE) and complex, high-performance models (e.g., Gradient Boosting) involves critical trade-offs:

Interpretability vs. Predictive Power: Simple models offer high interpretability, meaning their decisions can be easily understood and explained (e.g., a higher WoE value for a feature directly correlates with a higher likelihood of default). This is crucial for regulatory compliance, auditability, and explaining loan decisions to customers. However, they might sacrifice some predictive power, struggling to capture complex, non-linear relationships in the data, potentially leading to slightly lower accuracy. Complex models, conversely, often achieve superior predictive performance by learning intricate patterns, but they typically act as "black boxes," making their decision-making process opaque.

Regulatory Acceptance & Risk Management: Regulators often favor interpretable models due to the ease of validation, transparency, and the ability to conduct robust model risk management. They need to ensure models are fair, non-discriminatory, and accurately reflect the bank's risk exposure. Debugging and auditing are also simpler with transparent models. Complex models face stricter scrutiny, requiring advanced explainability techniques (like SHAP or LIME) which may still not fully satisfy regulatory demands for transparency. The inherent model risk is higher with complex models due to their opacity and potential for subtle biases.

Development & Maintenance: Simple models often require more intensive manual feature engineering (like WoE transformations), but their training and maintenance are generally straightforward. Complex models can automate much of this feature learning but might demand more computational resources for training and fine-tuning, and their behavior can be harder to troubleshoot.

For Bati Bank, balancing the need for strong predictive performance in identifying risky customers with the imperative of regulatory compliance and clear explainability for business operations is paramount. While a Gradient Boosting model might offer higher predictive accuracy, a Logistic Regression model enhanced with WoE might be a more pragmatic choice for its inherent transparency and ease of regulatory approval, especially when dealing with core credit decisions.

Future Work: CI/CD and Model Monitoring (Conceptual)
While the core model development and deployment are complete, a production-ready machine learning system requires robust CI/CD and continuous monitoring.

Continuous Integration/Continuous Deployment (CI/CD)
Goal: Automate the entire process from code changes to model deployment, ensuring consistent builds, testing, and rapid, reliable delivery.

Key Components:

Automated Testing: Implement unit, integration, and API tests for all code components (src/, app.py). These tests would run automatically on every code commit to ensure code quality and prevent regressions.

Dockerization: Package the Flask application, the trained model, and the data processing pipeline into a Docker image. This creates a portable and consistent environment for deployment.

Automated Deployment Workflows: Utilize CI/CD tools (e.g., GitHub Actions) to define workflows that automatically build and test the Docker image, push it to a container registry, and then deploy it to a production environment (e.g., a cloud VM or Kubernetes cluster) upon successful validation.

MLflow Integration: Leverage MLflow's tracking and model registry capabilities within the CI/CD pipeline. New model versions from training runs would be logged and registered, and the deployment process would fetch the desired model version directly from the MLflow Model Registry.

Model Monitoring
Goal: Continuously observe the model's performance and behavior in production to ensure it remains accurate and reliable over time.

Key Aspects:

Performance Monitoring: Track key classification metrics (Precision, Recall, F1-score for the high-risk class, ROC AUC, PR AUC) on live inference data. This requires a feedback loop to collect actual outcomes (e.g., customer default status). Dashboards and alerting systems would be set up to notify stakeholders of any performance degradation.

Data Drift Detection: Monitor the statistical properties and distributions of incoming production data features. Significant deviations from the training data distribution (data drift) can indicate that the model's assumptions are no longer valid, necessitating retraining.

Concept Drift Detection: Identify changes in the relationship between input features and the true target variable. This is more complex but crucial for long-term model relevance.

Automated Retraining Strategy: Define clear triggers for model retraining (e.g., scheduled intervals, performance degradation, detected data/concept drift, or availability of new labeled data). This retraining process would be integrated back into the CI/CD pipeline.