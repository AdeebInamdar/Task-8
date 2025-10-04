# Task 8: Unsupervised Customer Segmentation using K-Means Clustering 📊

This repository contains the solution for Task 8 of the AI & ML Internship, focusing on **unsupervised learning** through **K-Means Clustering**. The objective is to perform customer segmentation to identify natural groupings within the data.

## 🎯 Objective

The primary goal was to perform customer segmentation on a retail dataset to identify distinct customer groups based on spending habits and income, following a structured clustering and evaluation process.

## 💻 Tools and Libraries

* **Python 3.x**
* **Pandas:** For data loading and manipulation.
* **Scikit-learn (sklearn):** For implementing K-Means and calculating the Silhouette Score.
* **Matplotlib:** For data visualization (Elbow Method and final clusters).

## 🗄️ Dataset

* **Dataset Used:** Mall Customer Segmentation Dataset (`Mall_Customers.csv`).
* **Features for Clustering:** `Annual Income (k$)` and `Spending Score (1-100)`.

## ⚙️ Steps Performed (Mini Guide)

The following steps, as outlined in the task, were executed:

1.  **Data Loading and Selection:** Loaded the dataset and selected the two key numerical features for clustering.
2.  **Optimal K Determination (Elbow Method):** Used the **Elbow Method** (Inertia/WCSS vs. K) to find the ideal number of clusters.
3.  **K-Means Fitting:** Fitted the K-Means model using the determined optimal value of **$K=5$** and assigned cluster labels.
4.  **Cluster Evaluation:** Evaluated the quality of the clustering using the **Silhouette Score**.
5.  **Visualization:** Created scatter plots to visualize the separation of the five final clusters and their centroids.

***

## 📝 Results and Evaluation

### 1. Optimal K Determination

* **Conclusion:** The elbow occurred at **$K=5$**, indicating 5 clusters provide the best balance between complexity and variance reduction.
* **Visualization:** See the generated **`elbow_method.png`** file.

### 2. Clustering Performance

* **Optimal K Used:** 5
* **Silhouette Score:** **$\approx 0.5539$** (A score over 0.5 suggests the clusters are reasonably well-defined and separated.)

### 3. Customer Segments (Centroids)

The model identified five distinct customer segments based on their average income and spending profile:

| Segment | Avg. Annual Income (k\$) | Avg. Spending Score (1-100) | Interpretation (Marketing Segment) |
| :---: | :---: | :---: | :--- |
| **1** | $\approx 55.30$ | $\approx 49.52$ | **Average Customers** |
| **2** | $\approx 86.54$ | $\approx 82.13$ | **Target (High Income, High Spend)** - *The most valuable segment.* |
| **3** | $\approx 25.73$ | $\approx 79.36$ | **Careful Spenders (Low Income, High Spend)** |
| **4** | $\approx 88.20$ | $\approx 17.11$ | **Miser (High Income, Low Spend)** |
| **5** | $\approx 26.30$ | $\approx 20.91$ | **Frugal (Low Income, Low Spend)** |

### 4. Final Visualization

The final segmentation is visualized in **`kmeans_clusters.png`**.

***

## 🚀 How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-Link]
    ```
2.  **Ensure you have the necessary files:** The script file (e.g., `main.py`) and `Mall_Customers.csv` must be in the same directory.
3.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib
    ```
4.  **Execute the script:**
    ```bash
    python [Your-Script-Name].py
    ```
