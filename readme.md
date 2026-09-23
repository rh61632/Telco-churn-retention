# Telco Customer Churn Prediction with XGBoost & SHAP

An end-to-end machine learning project to predict customer attrition using the IBM Telco Customer Churn dataset. The project trains an optimized gradient boosting classifier (`XGBoost`), handles class imbalance dynamically, evaluates model discrimination via ROC-AUC and precision-recall metrics, and utilizes **SHAP (SHapley Additive exPlanations)** to interpret both global drivers and individual churn risk.

---

## 📌 Project Overview

Customer churn represents a critical revenue risk for subscription-based telecommunications providers. This solution delivers:
1. **Automated Feature Preprocessing:** Cleans non-numeric edge cases, imputes values, and one-hot encodes categorical dimensions.
2. **Feature Engineering:** Computes financial behavioral indicators such as `Charge_Ratio` ($\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1}$).
3. **Imbalance-Aware Classification:** Applies a dynamic `scale_pos_weight` factor inside an XGBoost classifier to avoid majority-class bias.
4. **Explainable AI (XAI):** Implements `shap.TreeExplainer` to produce Beeswarm, Bar, and Waterfall attribution plots for business stakeholders.
5. **Artifact Persistence:** Exports the trained model (`model.pkl`) and the exact feature schema (`model_columns.pkl`) for production scoring pipelines.

---

## 🗂️ Project Structure

```text
telco-churn/
├── telcolbm.csv            # IBM Telco customer raw dataset
├── main.ipynb              # End-to-end experimentation & analysis notebook
├── model.pkl               # Serialized trained XGBoost model
├── model_columns.pkl       # Saved feature column schema
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/telco-churn.git
   cd telco-churn
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Linux/macOS
   # .\venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies
* `xgboost`
* `shap`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`

---

## 🛠️ Pipeline Architecture

### 1. Data Cleaning & Feature Engineering
* **`TotalCharges` Sanitation:** Converted object data types with whitespace edge cases to numeric values via `pd.to_numeric(..., errors='coerce')`, filling zero-tenure NaNs with `0`.
* **Identifier Pruning:** Dropped non-predictive `customerID`.
* **Target Mapping:** Mapped `Churn` labels (`Yes` $\to 1$, `No` $\to 0$).
* **Ratio Engineering:**
  $$\text{Charge\_Ratio} = \frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1}$$
  This captures early customer onboarding friction vs. long-term account stability.
* **Encoding:** Applied dummy encoding (`pd.get_dummies(..., drop_first=True)`) to categorical columns.

### 2. Model Training & Imbalance Handling
* **Stratified Split:** 80/20 train/test split maintaining label distributions.
* **Class Weighting:** Automatically computed the negative-to-positive ratio:
  $$\text{scale\_pos\_weight} = \frac{\sum (y_{\text{train}} == 0)}{\sum (y_{\text{train}} == 1)}$$
* **Estimator Configuration:**
  ```python
  xgb.XGBClassifier(
      n_estimators=150,
      max_depth=4,
      learning_rate=0.05,
      scale_pos_weight=scale_pos_weight,
      subsample=0.8,
      random_state=42,
      eval_metric="logloss"
  )
  ```

---

## 📊 Model Evaluation & SHAP Interpretability

The model is evaluated on the holdout test set using ROC-AUC and detailed classification metrics (Precision, Recall, and F1-Score).

### Explainability Highlights (SHAP Analysis)
Using `shap.TreeExplainer(model)` on the holdout set yields key operational insights:

* **Top Protective Drivers (Reduces Churn):**
  * **`Contract_Two year` & `Contract_One year`:** Multi-year commitments drastically pull SHAP values down (negative log-odds impact), serving as the primary shield against churn.
  * **`tenure`:** High tenure heavily reduces departure risk.
  * **`OnlineSecurity` & `TechSupport`:** Customers with active add-on support exhibit higher retention rates.
* **Top Churn Accelerators (Increases Churn):**
  * **`Charge_Ratio`:** High ratios (typical of new, uncommitted accounts with high relative monthly spend) are strong positive drivers toward churn.
  * **`InternetService_Fiber optic`:** Fiber optic customers show consistently higher churn probabilities, suggesting potential pricing sensitivity or service friction.
  * **`PaymentMethod_Electronic check`:** Customers paying via manual electronic checks are notably more prone to attrition than automated payment users.

---

## 🚀 Serving Artifacts

Trained artifacts can be reloaded for downstream inference:

```python
import pickle
import pandas as pd

# Load artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Align inference data to training schema
# df_inference = df_inference.reindex(columns=model_columns, fill_value=0)
# predictions = model.predict_proba(df_inference)[:, 1]
```

---

## 📄 License
Distributed under the MIT License. Feel free to adapt and expand for your applications.