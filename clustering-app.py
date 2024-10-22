import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd

def generate_data(n_samples, centers, cluster_std):
    X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=centers, cluster_std=cluster_std, random_state=42)
    return pd.DataFrame(X, columns=["Feature 1", "Feature 2"])

def apply_kmeans(data, k):
    kmeans = KMeans(n_clusters=k)
    data['Cluster'] = kmeans.fit_predict(data)
    return data, kmeans.cluster_centers_

def apply_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Cluster'] = dbscan.fit_predict(data)
    return data

def main():
    st.title("Clustering Algorithm Parameter Selection")

    with st.sidebar:
        st.header("Dataset Parameters")
        
        n_samples = st.number_input("Number of samples", min_value=100, max_value=1000, value=300)
        centers = st.number_input("Number of centers", min_value=2, max_value=10, value=4)
        cluster_std = st.slider("Cluster Standard Deviation", min_value=0.1, max_value=3.0, step=0.1, value=0.60)

        st.header("Clustering Model Selection")
        
        model_choice = st.selectbox("Choose clustering model:", ("KMeans", "DBSCAN"))
        
        if model_choice == "KMeans":
            k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
        
        if model_choice == "DBSCAN":
            eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, step=0.1, value=0.5)
            min_samples = st.slider("MinPts", min_value=1, max_value=20, value=5)

    data = generate_data(n_samples, centers, cluster_std)

    if model_choice == "KMeans":
        clustered_data, centers = apply_kmeans(data, k)
        st.write(f"KMeans selected with K={k}")
    else:
        clustered_data = apply_dbscan(data, eps, min_samples)
        st.write(f"DBSCAN selected with eps={eps} and MinPts={min_samples}")

    fig, ax = plt.subplots()
    scatter = ax.scatter(clustered_data['Feature 1'], clustered_data['Feature 2'], c=clustered_data['Cluster'], cmap='viridis')
    if model_choice == "KMeans":
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label="Centers")
    plt.title(f"{model_choice} Clustering")
    st.pyplot(fig)

if __name__ == "__main__":
    main()

