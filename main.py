import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load and Prepare Data
df = pd.read_csv('Mall_Customers.csv')

# Select the features for clustering: 'Annual Income (k$)' and 'Spending Score (1-100)'
X = df.iloc[:, [3, 4]].values

print("--- Data Used for Clustering (Annual Income (k$), Spending Score (1-100)) ---")
# FIX: Using .to_string() to avoid the 'tabulate' dependency error.
print(pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)']).head().to_string(index=False))
print("\n" + "="*70 + "\n")

# 2. Elbow Method to find optimal K
wcss = []
k_range = range(1, 11)

for i in k_range:
    kmeans_model = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans_model.fit(X)
    wcss.append(kmeans_model.inertia_)

# Plot the Elbow Method graph and save it
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

print("--- Elbow Method Plot Saved as 'elbow_method.png' ---")
print("Visual inspection of the plot suggests K=5 as the optimal number of clusters.\n")
print("="*70 + "\n")

# 3. K-Means Clustering (using optimal K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 4. Evaluate clustering using Silhouette Score
silhouette_avg = silhouette_score(X, y_kmeans)

print(f"--- K-Means Results for K={optimal_k} ---")
print(f"Optimal K: {optimal_k}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print("\nCluster Centroids (Annual Income (k$), Spending Score (1-100)):")
# FIX: Using .to_string() to avoid the 'tabulate' dependency error.
print(pd.DataFrame(centers, columns=['Annual Income (k$)', 'Spending Score (1-100)']).to_string(index=False))
print("\n" + "="*70 + "\n")

# 5. Visualize clusters with color-coding and centroids
plt.figure(figsize=(12, 8))

# Scatter plot for each cluster
colors = ['purple', 'blue', 'green', 'cyan', 'magenta'] # Use a list of colors for distinct clusters
for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1],
                s=100, label=f'Cluster {i+1}', color=colors[i])

# Plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            s=300, c='red', marker='*', label='Centroids', edgecolors='black', linewidth=1.5)

plt.title(f'K-Means Clustering of Mall Customers (K={optimal_k})')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clusters.png')
plt.close()

print("--- Final Cluster Plot Saved as 'kmeans_clusters.png' ---")

