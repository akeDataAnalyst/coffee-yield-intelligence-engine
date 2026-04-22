# Coffee Yield Intelligence & Intervention Engine

## Description
This project is an end-to-end analytical system designed to improve data reliability and intervention targeting in smallholder coffee systems. By integrating satellite-derived environmental data with statistical and machine learning models, the system generates an objective "ground truth" estimate of yield and identifies high-impact intervention opportunities.

## Problem
Agricultural decision-making often relies on farmer self-reported data, which is prone to systematic recall bias. This leads to:
- Overestimation of yields and land size
- Misallocation of sustainability investments
- Inaccurate impact measurement

## Solution
I developed a data-driven pipeline that replaces subjective reporting with objective modeling:

- Integrated satellite climate data (rainfall, temperature)
- Engineered agro-climatic features (hydro-thermal indices, lag variables)
- Built a predictive yield model using XGBoost
- Quantified bias using statistical testing (paired T-test, effect size)
- Applied SHAP analysis to identify key yield drivers

## Recommendation
- Prioritize post-harvest infrastructure (drying beds) to improve quality and effective yield
- Apply correction factors to survey-based datasets to improve decision accuracy
- Target regions with high production but low quality conversion efficiency

## Tech Stack
- Python (Pandas, NumPy, SciPy)
- Machine Learning (XGBoost, SHAP)
- Data Source: NASA POWER API
- Visualization: Plotly, Seaborn
- Deployment: Streamlit
