-- SQL-style business analysis queries for customer churn and revenue risk

-- 1) Churn rate by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
    AVG(MonthlyCharges) AS avg_monthly_charge,
    SUM(MonthlyCharges * 6) AS six_month_revenue_at_risk
FROM customer_churn
GROUP BY Contract
ORDER BY churn_rate DESC;

-- 2) Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
    SUM(MonthlyCharges * 6) AS six_month_revenue_at_risk
FROM customer_churn
GROUP BY PaymentMethod
ORDER BY six_month_revenue_at_risk DESC;

-- 3) Tenure bucket analysis
SELECT
    CASE
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49-72 months'
    END AS tenure_bucket,
    COUNT(*) AS customers,
    AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate,
    AVG(MonthlyCharges) AS avg_monthly_charge
FROM customer_churn
GROUP BY tenure_bucket
ORDER BY customers DESC;

-- 4) Service mix and churn
SELECT
    InternetService,
    TechSupport,
    OnlineSecurity,
    COUNT(*) AS customers,
    AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END) AS churn_rate
FROM customer_churn
GROUP BY InternetService, TechSupport, OnlineSecurity
ORDER BY churn_rate DESC, customers DESC;

-- 5) High-value customers at risk
SELECT
    customerID,
    Contract,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
    (MonthlyCharges * 6) AS six_month_revenue_at_risk
FROM customer_churn
WHERE MonthlyCharges >= 80
  AND Contract = 'Month-to-month'
ORDER BY six_month_revenue_at_risk DESC;
