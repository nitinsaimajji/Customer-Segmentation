import streamlit as st
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.title("Customer Segmentation App")

uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])

if uploaded_file is not None:
    # Load data
    data = pd.read_excel(uploaded_file)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*n_init.*")
    # Display sample data
    st.subheader("Sample Data")
    st.write(data.head())

    # Perform clustering
    most_recent_date = data['InvoiceDate'].max()
    customer_df = data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (most_recent_date - x.max()).days,
                                                  'InvoiceNo': 'count',
                                                  'TotalPrice': 'sum'})

    customer_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'},
                        inplace=True)

    # Display pairplot
    st.subheader("Recency, Frequency, and Monetary")
    st.write(customer_df)
   

    # Scale the data
    norm_df = StandardScaler().fit_transform(customer_df)

    # Elbow Curve
    st.subheader("Elbow Curve for K-Means Clustering")
    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init=10)
        kmeans.fit(norm_df)
        ssd.append(kmeans.inertia_)
        st.write("For n_clusters={}, the Elbow score is {}".format(num_clusters, kmeans.inertia_))

    fig = px.line(x=range_n_clusters, y=ssd,
                  title="Elbow Curve for K-Means Clustering",
                  labels={'x': 'Number of Clusters', 'y': 'Sum of Squared Distances (SSD)'})
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.plotly_chart(fig)

    # Silhouette analysis
    st.subheader("Silhouette Analysis")
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init=10)
        kmeans.fit(norm_df)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(norm_df, cluster_labels)
        st.write("For n_clusters={}, the silhouette score is {}".format(num_clusters, silhouette_avg))

    # Final KMeans clustering
    final_kmeans = KMeans(n_clusters=3, random_state=42)
    final_kmeans.fit(norm_df)

    final_df = pd.DataFrame(customer_df, columns=customer_df.columns, index=customer_df.index)
    final_df['Cluster'] = final_kmeans.labels_ + 1

    st.subheader("Cluster Statistics")
    st.write(final_df.groupby('Cluster').agg({'Monetary': 'mean',
                                              'Frequency': 'mean',
                                              'Recency': 'mean'}))

    
    grouped_data = final_df.groupby('Cluster')
    cluster_data = grouped_data.get_group(3)
    c3 = cluster_data.index
    c3 = list(c3)
    filtered_df = data[data["CustomerID"].isin(c3)]  # Corrected: use 'data' instead of 'df'

    st.subheader("Filtered Data for Cluster 3")
    st.write(filtered_df)
    st.write("Size of the filtered data:", filtered_df.size)

    # Additional Visualizations
    st.subheader("Bar Chart of Country Distribution")
    fig, ax = plt.subplots()
    countries = filtered_df['Country'].unique()
    counts = filtered_df['Country'].value_counts().sort_values(ascending=False)
    ax.bar(countries, counts)
    ax.set_title("Bar Chart of Country Distribution")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of People")
    st.pyplot(fig)

    # Bar Chart for Top 5 products mostly ordered
    st.subheader("Bar Chart for Top 5 products mostly ordered")
    counts = filtered_df['Description'].unique()
    counts_list = counts.tolist()
    counts_dict = {}
    for item in counts_list:
        count = filtered_df['Description'].value_counts().get(item, 0)
        counts_dict[item] = count
    counts_df = pd.DataFrame(list(counts_dict.items()), columns=['Description', 'Count'])
    counts_df = counts_df.sort_values(by='Count', ascending=False)
    c_df = counts_df.head()
    description = c_df['Description'].unique()
    count = c_df['Count']
    fig, ax = plt.subplots()
    ax.bar(description, count)
    ax.set_title("Bar Chart for Top 5 products mostly ordered")
    ax.set_xlabel("Description")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(description)))
    ax.set_xticklabels(description, rotation=45, ha="right")
    st.pyplot(fig)

   # Bar Chart for Total Price per Customer
    st.subheader("TOTAL AMOUNT SPENT BY EACH CUSTOMER")
    c = filtered_df.groupby('CustomerID').agg({'TotalPrice': 'sum'})
    plt.figure(figsize=(10, 6))
    sns.barplot(y='TotalPrice', x='CustomerID', data=c, palette='viridis')
    plt.xlabel('CustomerID')
    plt.ylabel('TotalPrice')
    plt.title('CustomerID vs. Total Price')
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

    # Monthly Sales
    st.subheader("MONTHLY SALES")
    filtered_df['YearMonth'] = filtered_df['InvoiceDate'].dt.to_period('M')
    monthly_total_sales = filtered_df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
    monthly_total_sales['YearMonth'] = monthly_total_sales['YearMonth'].astype(str)
    yearmonth = monthly_total_sales['YearMonth']
    total_price = monthly_total_sales['TotalPrice']
    plt.figure(figsize=(10, 6))
    plt.plot(yearmonth, total_price, marker='o', linestyle='-')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Price')
    plt.title('Total Price Over Time')
    plt.xticks(rotation=45)
    st.pyplot()

    # Yearly Sales
    st.subheader("YEARLY SALES")
    filtered_df['Year'] = filtered_df['InvoiceDate'].dt.year
    yearly_total_sales = filtered_df.groupby('Year')['TotalPrice'].sum().reset_index()
    st.write(yearly_total_sales)