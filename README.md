# Credit Risk Probability Model for Alternative Data

## Project Overview

This project focuses on building an end-to-end credit risk probability model for Bati Bank, a leading financial service provider. The goal is to develop a buy-now-pay-later service by creating a Credit Scoring Model that leverages alternative data provided by an upcoming successful eCommerce company.

Traditionally, credit scoring models rely on past borrower behavior. Our key innovation lies in transforming raw eCommerce transaction data into a predictive risk signal. By analyzing customer **Recency, Frequency, and Monetary (RFM) patterns**, we'll engineer a proxy for credit risk. This allows us to train a model that outputs a **risk probability score**, a vital metric that will inform loan approvals and terms for new customers.

### Key Objectives:

* **Define a proxy variable** for categorizing users as high-risk ("bad") or low-risk ("good").
* **Select observable features** that are strong predictors of the defined risk variable.
* **Develop a model** to assign a risk probability for new customers.
* **Develop a model** to assign a credit score from risk probability estimates.
* **Develop a model** to predict the optimal amount and duration of a loan.


## Project Structure

The project follows a standardized structure to ensure maintainability, modularity, and scalability:
credit-risk-model/
├── .github/workflows/ci.yml    # GitHub Actions workflows for CI/CD (linting, tests)
├── data/                       # Data storage (ignored by Git)
│   ├── raw/                    # Raw, untransformed source data
│   └── processed/              # Cleaned and processed data ready for model training
├── notebooks/
│   └── 1.0-eda.ipynb           # Jupyter notebooks for exploratory data analysis and ad-hoc experiments
├── src/                        # Source code for the application
│   ├── init.py             # Python package initialization file
│   ├── data_processing.py      # Script for data loading, cleaning, feature engineering, and target variable creation
│   ├── train.py                # Script for model training, evaluation, hyperparameter tuning, and MLflow tracking
│   ├── predict.py              # Script containing core prediction logic (used by the API)
│   └── api/                    # API related files
│       ├── main.py             # FastAPI application for exposing the model via an API endpoint
│       └── pydantic_models.py  # Pydantic models for API request/response validation and serialization
├── tests/
│   └── test_data_processing.py # Unit tests for critical functions, starting with data processing
├── Dockerfile                  # Defines the Docker image for containerizing the FastAPI application
├── docker-compose.yml          # Defines multi-container Docker applications for easy local setup/testing
├── requirements.txt            # Lists all Python package dependencies for the project
├── .gitignore                  # Specifies files and directories that Git should ignore
└── README.md                   # Project overview, setup guide, and key documentation

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.9+**: The primary programming language for this project.
* **Git**: For version control and managing the repository.
* **Docker**: Essential for containerizing the application and facilitating deployment.

### Setup Instructions

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ybtry/credit-risk-model.git](https://github.com/Ybtry/credit-risk-model.git)
    cd credit-risk-model
    ```
2.  **Create and activate a Python virtual environment:**
    It's crucial to use a virtual environment to manage project dependencies isolation.
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    Once your virtual environment is active, install all required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Data Acquisition

The raw transaction data (`transactions.csv`) required for this challenge can be obtained from the Kaggle platform:

* **Download from Kaggle:** [Xente Challenge | Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)
* **Placement:** After downloading, place the `transactions.csv` file into the `data/raw/` directory within your project structure.

---

## Credit Scoring Business Understanding

This section provides critical context on the financial industry's approach to credit risk and the strategic decisions driving our model's development.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord mandates a robust framework for financial institutions to assess and manage credit risk, particularly under its Pillar 1 (Minimum Capital Requirements) and Pillar 2 (Supervisory Review Process). This emphasis directly drives the need for an **interpretable and well-documented model**. Regulatory bodies require a clear understanding of how models arrive at their risk assessments to ensure compliance, prevent systemic risks, and validate capital adequacy. An interpretable model allows Bati Bank to explain its credit decisions to regulators, auditors, and even customers, fostering transparency and trust. Furthermore, it facilitates rigorous **model validation**, **stress testing**, and the identification of potential biases or flaws, all of which are critical for effective risk management and maintaining the bank's financial stability in a regulated environment. Without clear documentation and interpretability, the model would operate as a "black box," making it impossible to satisfy regulatory scrutiny or explain loan outcomes.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

Creating a proxy variable for "default" is necessary because the provided eCommerce dataset lacks an explicit label indicating whether a customer has defaulted on a loan or service. To build a supervised machine learning model, a target variable (our "ground truth") is indispensable. By defining "high-risk" customers based on observed behavioral patterns like Recency, Frequency, and Monetary (RFM) values, we infer a proxy for credit risk.

However, relying on a proxy variable introduces significant **business risks**. The primary risk is **misclassification**. If our proxy does not perfectly align with actual credit default, we face:
1.  **False Positives (Type I Error):** Classifying a truly creditworthy customer as high-risk. This leads to denying credit to good customers, resulting in lost revenue opportunities for Bati Bank and potentially damaging customer relationships.
2.  **False Negatives (Type II Error):** Classifying a truly high-risk customer as low-risk. This results in approving loans that are likely to default, leading to direct financial losses for the bank due to unrecovered principal and interest.
Beyond direct losses, making predictions based on an imperfect proxy can lead to **model drift** (as the relationship between the proxy and true default may evolve) and **unintended biases** if the proxy definition inadvertently correlates with protected characteristics. Ultimately, the model's effectiveness in mitigating true credit risk is contingent on how well this engineered proxy genuinely reflects actual default behavior.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial context, the choice between simple, interpretable models (e.g., Logistic Regression with Weight of Evidence - WoE) and complex, high-performance models (e.g., Gradient Boosting) involves critical trade-offs:

* **Interpretability vs. Predictive Power:** Simple models offer **high interpretability**, meaning their decisions can be easily understood and explained (e.g., a higher WoE value for a feature directly correlates with a higher likelihood of default). This is crucial for regulatory compliance, auditability, and explaining loan decisions to customers. However, they might sacrifice some **predictive power**, struggling to capture complex, non-linear relationships in the data, potentially leading to slightly lower accuracy. Complex models, conversely, often achieve **superior predictive performance** by learning intricate patterns, but they typically act as "black boxes," making their decision-making process opaque.

* **Regulatory Acceptance & Risk Management:** Regulators often favor interpretable models due to the ease of validation, transparency, and the ability to conduct robust model risk management. They need to ensure models are fair, non-discriminatory, and accurately reflect the bank's risk exposure. Debugging and auditing are also simpler with transparent models. Complex models face stricter scrutiny, requiring advanced explainability techniques (like SHAP or LIME) which may still not fully satisfy regulatory demands for transparency. The inherent model risk is higher with complex models due to their opacity and potential for subtle biases.

* **Development & Maintenance:** Simple models often require more intensive manual feature engineering (like WoE transformations), but their training and maintenance are generally straightforward. Complex models can automate much of this feature learning but might demand more computational resources for training and fine-tuning, and their behavior can be harder to troubleshoot.

For Bati Bank, balancing the need for strong predictive performance in identifying risky customers with the imperative of regulatory compliance and clear explainability for business operations is paramount. While a Gradient Boosting model might offer higher predictive accuracy, a Logistic Regression model enhanced with WoE might be a more pragmatic choice for its inherent transparency and ease of regulatory approval, especially when dealing with core credit decisions.
