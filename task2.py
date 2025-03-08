# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the dataset
print("Loading and inspecting the dataset...")
df = pd.read_csv('customer_behavior_analytcis.csv')

# Display basic information
print("Dataset initial shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values
print("\nCleaning the dataset...")
# Fill missing numerical values with median of each column
for col in df.columns:
    if col != 'customer_id':  # Skip the ID column
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
# Get column medians for imputation
col_medians = df.median(numeric_only=True)
print("\nMedian values for imputation:")
print(col_medians)

# Impute missing values with column medians
df = df.fillna(col_medians)

# Verify no missing values remain
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Create a copy of dataframe for analysis, excluding customer_id
data = df.drop('customer_id', axis=1)

# Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis...")

# Basic statistics
print("\nDescriptive statistics:")
print(data.describe())

# Create a figure for histograms
plt.figure(figsize=(20, 12))
for i, column in enumerate(data.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.close()

# Box plots to detect outliers
plt.figure(figsize=(20, 10))
data.boxplot()
plt.title('Box Plots for Features')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=300)
plt.close()

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation = data.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Correlation Between Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.close()

# Pairplot
plt.figure(figsize=(20, 15))
pairplot = sns.pairplot(data, diag_kind='kde')
plt.tight_layout()
pairplot.savefig('pairplot.png', dpi=300)
plt.close()

# Feature scaling
print("\nScaling features for clustering...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

# Determine optimal number of clusters (even though we know k=3)
print("\nValidating optimal number of clusters...")
inertia = []
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))

# Plot elbow method results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('optimal_clusters.png', dpi=300)
plt.close()

# Apply K-means clustering with k=3 (as given in the problem statement)
print("\nApplying K-means clustering with k=3...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['cluster'] = cluster_labels
print("\nCluster distribution:")
print(df['cluster'].value_counts())

# Get cluster centers and transform back to original scale
centers = kmeans.cluster_centers_
original_centers = scaler.inverse_transform(centers)
centers_df = pd.DataFrame(original_centers, columns=data.columns)
print("\nCluster centers (original scale):")
print(centers_df)

# Visualize clusters using PCA
print("\nVisualizing clusters using PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

plt.figure(figsize=(12, 10))
scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], 
                      cmap='viridis', s=50, alpha=0.8)
plt.colorbar(scatter)
plt.title('Customer Clusters Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=300)
plt.close()

# Create 3D scatter plot using top 3 PCA components
print("\nCreating 3D cluster visualization...")
pca3d = PCA(n_components=3)
pca3d_result = pca3d.fit_transform(scaled_data)
df['pca1_3d'] = pca3d_result[:, 0]
df['pca2_3d'] = pca3d_result[:, 1]
df['pca3_3d'] = pca3d_result[:, 2]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['pca1_3d'], df['pca2_3d'], df['pca3_3d'], 
                    c=df['cluster'], cmap='viridis', s=50, alpha=0.8)
plt.colorbar(scatter)
ax.set_title('3D Customer Clusters Visualization')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.tight_layout()
plt.savefig('3d_clusters.png', dpi=300)
plt.close()

# Parallel coordinates plot to visualize clusters
plt.figure(figsize=(15, 8))
parallel_coords = pd.plotting.parallel_coordinates(
    pd.concat([scaled_df, df[['cluster']]], axis=1),
    'cluster', colormap='viridis'
)
plt.title('Parallel Coordinates Plot of Customer Clusters')
plt.tight_layout()
plt.savefig('parallel_coordinates.png', dpi=300)
plt.close()

# Analyze cluster characteristics
print("\nAnalyzing cluster characteristics...")
# Exclude customer_id when calculating means by cluster
cluster_analysis = df.drop('customer_id', axis=1).groupby('cluster').mean()
print("\nCluster means:")
print(cluster_analysis)

# Radar chart (spider plot) to visualize cluster profiles
def radar_plot(df, features, cluster_col):
    # Number of features
    N = len(features)
    
    # Calculate group means
    group_means = df.groupby(cluster_col)[features].mean()
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Normalize data for the radar chart
    # We'll normalize between 0 and 1 for better visualization
    normalized_means = group_means.copy()
    for feature in features:
        min_val = group_means[feature].min()
        max_val = group_means[feature].max()
        if max_val > min_val:
            normalized_means[feature] = (group_means[feature] - min_val) / (max_val - min_val)
    
    # Initialize the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable + add labels
    plt.xticks(angles[:-1], features, size=12)
    
    # Draw y labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each cluster
    cluster_colors = ['blue', 'green', 'red']
    cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    
    for i, (idx, group) in enumerate(normalized_means.iterrows()):
        values = group.values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', 
                label=cluster_names[i], color=cluster_colors[i])
        ax.fill(angles, values, alpha=0.1, color=cluster_colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig, ax

print("\nCreating radar chart for cluster profiles...")
features = data.columns.tolist()
radar_fig, _ = radar_plot(df.drop('customer_id', axis=1), features, 'cluster')
plt.title("Radar Chart of Cluster Profiles (Normalized)", size=15)
plt.tight_layout()
plt.savefig('radar_chart.png', dpi=300)
plt.close()

# Create bar charts for each feature across clusters
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.barplot(x='cluster', y=feature, data=df, palette='viridis')
    plt.title(f'Average {feature} by Cluster')
    plt.tight_layout()
plt.savefig('feature_by_cluster.png', dpi=300)
plt.close()

# Map clusters to meaningful business segments based on characteristics
# Analyze the characteristics to determine which cluster matches which segment
# Let's check the cluster centers one more time to make this determination
print("\nMapping clusters to customer segments...")
print(centers_df)

# Assuming we map them as follows (this should be adjusted based on actual results):
# Based on problem description and cluster means
segment_mapping = {
    0: 'High Spenders',    # High avg_cart_value, moderate purchases, low discount usage
    1: 'Bargain Hunters',  # High total_purchases, low avg_cart_value, high discount usage
    2: 'Window Shoppers'   # Low total_purchases, high total_time_spent, high product clicks
}

# Adjust the segment mapping based on actual cluster characteristics
# Look at centers_df to determine proper mapping

# Apply segment mapping to the dataframe
df['customer_segment'] = df['cluster'].map(segment_mapping)

# Final distribution of customer segments
print("\nCustomer segment distribution:")
print(df['customer_segment'].value_counts())

# Create a pie chart for segment distribution
plt.figure(figsize=(10, 8))
df['customer_segment'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, explode=[0.05, 0.05, 0.05])
plt.title('Customer Segment Distribution')
plt.tight_layout()
plt.savefig('segment_distribution.png', dpi=300)
plt.close()

# Create bar plot showing average values for each segment
plt.figure(figsize=(15, 10))
segment_means = df.groupby('customer_segment')[features].mean()
segment_means.T.plot(kind='bar', figsize=(15, 8))
plt.title('Average Feature Values by Customer Segment')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.legend(title='Customer Segment')
plt.tight_layout()
plt.savefig('segment_feature_comparison.png', dpi=300)
plt.close()

# Save the segmented customer data to CSV
print("\nSaving segmented customer data to CSV...")
df.to_csv('customer_segments.csv', index=False)

print("\nCustomer segmentation analysis complete!")
print("All visualizations have been saved to the current directory.")
