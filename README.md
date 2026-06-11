# Healthcare Cost Estimator

Machine learning regression project for estimating annual healthcare insurance charges from age, BMI, smoking status, gender, dependents, and region.

## Project Contents

- `app/app.py` - Streamlit application with dashboard insights, prediction, and plots/data tabs.
- `data/insurance.csv` - Original dataset.
- `data/cleaned_insurance.csv` - Cleaned dataset.
- `data/insurance_processed.csv` - Encoded and feature-engineered dataset.
- `data/train.csv` and `data/test.csv` - Training and testing splits.
- `models/healthcare_cost_model.pkl` - Trained healthcare cost model.
- `models/feature_names.pkl` - Feature schema used by the model.
- `reports/EDA_Report.docx` - Exploratory data analysis report.
- `reports/Model_Evaluation_Report.docx` - Regression model evaluation report.
- `reports/Feature_Importance_Report.docx` - Feature importance report.
- `reports/Healthcare_Cost_Estimator_Documentation.docx` - Final project documentation.
- `Documentation/` - Copies of the final DOCX documents.

## Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app/app.py
```

## Model Summary

The saved model is an XGBoost regressor. The recorded comparison table in `reports/model_comparison.csv` selects Tuned XGBoost as the best model by R2 score.

Predictions are returned in USD because the dataset target column, `charges`, is recorded in USD.
