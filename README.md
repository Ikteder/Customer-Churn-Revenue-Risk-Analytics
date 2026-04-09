# Customer Churn and Revenue Risk Analytics

A GitHub-ready data science project that predicts customer churn, quantifies revenue risk, and translates model outputs into retention actions.

## Project goals

- Predict which customers are likely to churn
- Identify the strongest churn drivers
- Estimate which customer segments create the highest revenue risk
- Recommend retention actions grounded in model outputs and business analysis

## Tech stack

- Python
- pandas
- scikit-learn
- XGBoost
- matplotlib / seaborn
- SQL-style business analysis
- Streamlit dashboard

## Repository structure

```text
customer-churn-revenue-risk-analytics/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── telco_churn.csv           # add dataset here
│   └── processed/
├── notebooks/
│   └── churn_analysis_starter.ipynb
├── reports/
│   ├── figures/
│   └── metrics_summary.csv
├── sql/
│   └── churn_analysis.sql
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── business_analysis.py
│   └── utils.py
├── tests/
│   └── test_feature_engineering.py
├── .gitignore
├── README.md
├── requirements.txt
└── run_project.py
```

## Dataset

Recommended public dataset:
- IBM Telco Customer Churn

Place the CSV file in:

```text
data/raw/telco_churn.csv
```

## How to run

```bash
pip install -r requirements.txt
python run_project.py
streamlit run app/streamlit_app.py
```

## What the pipeline does

1. Loads and cleans customer data
2. Handles missing values and basic encoding
3. Builds engineered features such as tenure bucket and monthly spend bands
4. Trains Logistic Regression, Random Forest, XGBoost, and SVM
5. Evaluates ROC-AUC, F1, precision, recall, and confusion matrix
6. Produces business summaries and revenue-risk segmentation
7. Saves metrics and plots to `reports/`

## Results section for GitHub

The figure panels below are pre-wired into the README. Replace them with real generated charts after you run the project.

<p align="center">
  <img src="reports/figures/churn_by_contract.png" width="48%" />
  <img src="reports/figures/churn_by_tenure_bucket.png" width="48%" />
</p>
<p align="center">
  <img src="reports/figures/model_comparison.png" width="48%" />
  <img src="reports/figures/feature_importance_top15.png" width="48%" />
</p>
<p align="center">
  <img src="reports/figures/revenue_risk_by_segment.png" width="48%" />
  <img src="reports/figures/confusion_matrix_best_model.png" width="48%" />
</p>

## Business questions this project answers

- Who is most likely to churn?
- Why are they churning?
- Which customer segments represent the highest revenue risk?
- What action should the business prioritize first?

## Deliverables

- cleaned dataset pipeline
- SQL queries and pandas summaries
- training script
- evaluation report
- dashboard starter
- feature importance analysis

## Suggested talking points for interviews

- Framed churn prediction as both an ML and business decision-support problem
- Compared multiple models instead of relying on a single baseline
- Connected churn probability to revenue risk prioritization
- Built a reproducible repo with analytics, modeling, plots, and a dashboard
