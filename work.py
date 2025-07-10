# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(
    page_title="Mall Customer Segmentation App",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title of the web application
st.title(" Mall Customer Segmentation")
st.markdown("""
This application segments mall customers using **K-Means Clustering**.  
Upload your dataset, choose number of clusters, and view the results visually.
""")

# Sidebar for user inputs
st.sidebar.header(" Configuration")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload the Mall Customers CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    st.subheader(" Raw Data Preview")
    st.dataframe(data.head())

    # Show basic information about dataset
    st.markdown("####  Dataset Summary")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    st.write(data.describe())

    # Select features for clustering
    st.markdown("####  Feature Selection for Clustering")
    features = st.multiselect(
        "Select 2 numerical features to apply K-Means clustering",
        options=data.select_dtypes(include='number').columns.tolist(),
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(features) != 2:
        st.warning("Please select exactly 2 features for 2D visualization.")
    else:
        # Scale the selected features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features])

        # Choose number of clusters
        k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Add cluster labels to original dataframe
        data['Cluster'] = cluster_labels

        st.markdown("####  Clustered Data")
        st.dataframe(data[[*features, 'Cluster']].head())

        # Visualize clusters using a scatter plot
        st.markdown("####  Cluster Visualization")

        fig, ax = plt.subplots()
        palette = sns.color_palette("bright", k)
        sns.scatterplot(
            x=scaled_features[:, 0], 
            y=scaled_features[:, 1], 
            hue=cluster_labels, 
            palette=palette,
            ax=ax
        )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title("K-Means Clustering")
        st.pyplot(fig)

        # Show cluster sizes
        st.markdown("####  Cluster Sizes")
        st.bar_chart(data['Cluster'].value_counts().sort_index())

        # Conclusion
        st.markdown("###  Summary")
        st.write(f"The customers have been grouped into {k} clusters based on their {features[0]} and {features[1]}.")
        st.write("This segmentation can help the mall in targeted marketing strategies.")

else:
    st.info("Please upload the dataset to start.")
