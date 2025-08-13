import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Rename columns for easier use
df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Score'
}, inplace=True)

# Plot 1: Raw data (Before clustering)
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Income', y='Score', data=df)
plt.title('Customer Distribution - Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()

# Prepare data
X = df[['Income', 'Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Plot 2: K-Means Clustering
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Income', y='Score', hue='KMeans_Cluster', data=df, palette='Set1', s=100)
plt.title('K-Means Clustering - Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Plot 3: DBSCAN Clustering
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Income', y='Score', hue='DBSCAN_Cluster', data=df, palette='Set2', s=100)
plt.title('DBSCAN Clustering - Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Plot 4: Side-by-Side Comparison
plt.figure(figsize=(14, 6))

# K-Means subplot
plt.subplot(1, 2, 1)
sns.scatterplot(x='Income', y='Score', hue='KMeans_Cluster', data=df, palette='Set1', s=100)
plt.title('K-Means Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.grid(True)

# DBSCAN subplot
plt.subplot(1, 2, 2)
sns.scatterplot(x='Income', y='Score', hue='DBSCAN_Cluster', data=df, palette='Set2', s=100)
plt.title('DBSCAN Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# Cluster Summaries
print("K-Means Clusters Summary:")
print(df.groupby('KMeans_Cluster')[['Income', 'Score']].mean())

print("\nDBSCAN Clusters Summary (excluding noise):")
print(df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[['Income', 'Score']].mean())

print(f"\nDBSCAN Noise Points: {(df['DBSCAN_Cluster'] == -1).sum()}")
