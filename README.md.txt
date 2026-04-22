coffee-yield-intelligence-engine

# Coffee Yield Intelligence & Intervention Engine
**Developed by Aklilu Abera | Supply Chain Data Analyst**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](YOUR_STREAMLIT_URL_HERE)

## Description
This project is an end-to-end analytical suite designed to identify, quantify, and correct data integrity gaps in the Ethiopian coffee supply chain. By integrating NASA POWER satellite data with simulated ECX (Ethiopia Commodity Exchange) trade ledgers, the engine builds a "Ground Truth" model to bypass unreliable farmer recall data and simulate the impact of infrastructure-led interventions.

## The Problem: The Recall Bias Gap
In agricultural supply chains, data often relies on farmer surveys. However, human memory is subject to Systematic Recall Bias. 
* **The Variance:** Industry research indicates that farmer recall often overestimates yield or land size by 20–30%.
* **Statistical Audit:** This project performed a paired T-test and calculated a Cohen's d of 1.78, proving a Critical Bias that leads to over-allocation of intervention budgets and skewed impact reporting.

## The Solution: A Data-Driven Truth Engine
The solution moves away from subjective surveys toward an objective, multi-layered machine learning approach:

1.  **Environmental Ground Truth:** Automated ingestion of daily rainfall and temperature data via the NASA POWER API for specific coordinates (Jimma, Sidama, Yirgacheffe).
2.  **Feature Engineering:** Developed "Hydro-Thermal" indices and biological lag features to capture the relationship between climate and coffee physiology.
3.  **Machine Learning (XGBoost):** Built a Gradient Boosted model to predict actual yield volumes. While weather explains 17.5% of variance (the Climate Ceiling), the model isolates the remaining variance for on-farm management.
4.  **SHAP Interpretation:** Utilized Game Theory (SHAP) to rank drivers, identifying Cumulative Rainfall and Quality Grade as the primary signals for throughput.

## Recommendation & Strategic Impact
Based on the Intervention Impact Simulation, the following strategic recommendations were identified:

* **Infrastructure over Information:** A simulated 1-point improvement in post-harvest quality (Grade) results in a 7.2% Yield Lift in measurable supply.
* **Targeted Investment:** Prioritize drying-bed infrastructure in the Jimma Region, which currently shows the highest gap between production volume and quality success rate.
* **Operational Correction:** Enveritas should apply the calculated 1.25x Correction Factor to all raw farmer-survey data to align field reports with actual export capacity.

---

## Tech Stack
* **Data Ingestion:** NASA POWER API (REST)
* **Analysis:** Python (Pandas, NumPy, Scipy)
* **Machine Learning:** XGBoost, SHAP (Explainable AI)
* **Visualization:** Plotly, Seaborn
* **Deployment:** Streamlit (Dark Mode Executive Dashboard)
