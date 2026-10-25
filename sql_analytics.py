
import pandas as pd
import pymysql
import streamlit as st
import plotly.express as px
from getpass import getpass

st.title("🗄️ MySQL Churn Analytics")

st.write(
    "Explore customer churn metrics retrieved directly from MySQL."
)

# Database connection settings
db_user = st.sidebar.text_input(
    "MySQL username", value="root"
)
db_password = st.sidebar.text_input(
    "MySQL password", type="password"
)
db_host = st.sidebar.text_input(
    "MySQL host", value="localhost"
)

if st.sidebar.button("Load SQL Analytics"):
    try:
        connection = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database="customer_churn_db",
            charset="utf8mb4"
        )

        queries = {
            "Contract": """
                SELECT Contract AS Segment,
                    COUNT(*) AS Total_Customers,
                    SUM(Churn = 'Yes') AS Churned_Customers,
                    ROUND(100 * AVG(Churn = 'Yes'), 2)
                        AS Churn_Rate
                FROM customers
                GROUP BY Contract
                ORDER BY Churn_Rate DESC
            """,
            "Payment Method": """
                SELECT PaymentMethod AS Segment,
                    COUNT(*) AS Total_Customers,
                    SUM(Churn = 'Yes') AS Churned_Customers,
                    ROUND(100 * AVG(Churn = 'Yes'), 2)
                        AS Churn_Rate
                FROM customers
                GROUP BY PaymentMethod
                ORDER BY Churn_Rate DESC
            """,
            "Internet Service": """
                SELECT InternetService AS Segment,
                    COUNT(*) AS Total_Customers,
                    SUM(Churn = 'Yes') AS Churned_Customers,
                    ROUND(100 * AVG(Churn = 'Yes'), 2)
                        AS Churn_Rate
                FROM customers
                GROUP BY InternetService
                ORDER BY Churn_Rate DESC
            """
        }

        for title, query in queries.items():
            df = pd.read_sql(query, connection)

            st.subheader(f"Churn by {title}")
            st.dataframe(df, use_container_width=True, hide_index=True)

            fig = px.bar(
                df,
                x="Segment",
                y="Churn_Rate",
                text="Churn_Rate",
                title=f"Churn Rate by {title}",
                labels={
                    "Segment": title,
                    "Churn_Rate": "Churn Rate (%)"
                }
            )
            fig.update_traces(
                texttemplate="%{text}%",
                textposition="outside"
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        connection.close()
        st.success("SQL analytics loaded successfully!")

    except Exception as error:
        st.error(f"Could not load SQL analytics: {error}")
