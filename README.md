

\# Customer Churn Prediction \& Retention Intelligence Platform



\## Overview

An end-to-end machine learning project that predicts customer churn risk and analyzes customer retention patterns using Python, Random Forest, MySQL, and Streamlit.



\## Features

\- Customer churn probability prediction

\- Interactive Streamlit prediction dashboard

\- Business KPIs and churn visualizations

\- MySQL analytics by contract, payment method, and internet service

\- Random Forest feature importance analysis

\- Model comparison and evaluation reports



\## Technology Stack

\- Python, pandas, scikit-learn

\- Random Forest and Logistic Regression

\- MySQL and PyMySQL

\- Streamlit and Plotly

\- Matplotlib and joblib



\## Dataset

IBM Telco Customer Churn dataset:

\- 7,043 customers

\- 1,869 churned customers

\- Historical churn rate: 26.54%



\## Model Performance

Random Forest test-set results:



| Metric | Score |

|---|---:|

| Accuracy | 76.51% |

| Precision | 54.62% |

| Recall | 67.91% |

| F1-score | 60.55% |

| ROC-AUC | 83.64% |



The classification threshold is 0.55, selected using validation data.



\## SQL Insights



| Segment | Churn Rate |

|---|---:|

| Month-to-month contracts | 42.71% |

| One-year contracts | 11.27% |

| Two-year contracts | 2.83% |

| Electronic check users | 45.29% |

| Fiber optic users | 41.89% |



These figures describe historical associations and do not prove causation.



\## Run the Applications



Install dependencies:

```bash

pip install pandas scikit-learn matplotlib seaborn joblib streamlit plotly pymysql

```



Prediction dashboard:

```bash

streamlit run app.py

```



SQL analytics:

```bash

streamlit run sql\_analytics.py

```



Model explainability:

```bash

streamlit run model\_explainability.py

```



\## Limitations

Predictions are estimates, not guarantees. Feature importance does not prove causation. Model performance may differ on new datasets.



\## Future Improvements

\- SHAP-based explanations

\- Model monitoring

\- Automated tests

\- Cloud deployment

\- Integrated SQL analytics and prediction dashboard



