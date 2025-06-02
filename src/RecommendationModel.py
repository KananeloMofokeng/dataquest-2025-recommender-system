# Customer Segmentation Analysis Pipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import psutil  # For memory monitoring
from tqdm import tqdm  # For progress bars
from joblib import dump  # For model saving

# Constants
INPUT_PATH = r'C:/Users/nyaka/Downloads/DATAQUEST/dq_recsys_challenge_2025_cleaned.csv'
OUTPUT_PATH = r'C:/Users/nyaka/Downloads/DATAQUEST/TRY9112221111eeee11444.csv'
RANDOM_STATE = 42
CLUSTER_RANGE = (2, 10)
SAMPLE_FRAC = 0.3
COLORS = sns.color_palette("Set2")
RECOMMENDATION_CHUNK_SIZE = 50000  # Process customers in chunks for memory efficiency

# Helper functions
def load_and_preprocess():
    """Load and preprocess raw interaction data"""
    print("Loading raw data...")
    df = pd.read_csv(INPUT_PATH, encoding='utf-8', sep=';')
    df['int_date'] = pd.to_datetime(df['int_date'])

    # Create interaction flags
    df['interaction'] = df['interaction'].str.lower()
    df['checkout_flag'] = df['interaction'] == 'checkout'
    df['click_flag'] = df['interaction'] == 'click'

    # Aggregate to customer-level features
    print("Aggregating customer data...")
    customer_data = df.groupby('idcol').agg(
        recency_days=('int_date', lambda x: (df['int_date'].max() - x.max()).days),
        interaction_count=('interaction', 'count'),
        checkout_count=('checkout_flag', 'sum'),
        click_count=('click_flag', 'sum'),
        item_type_diversity=('item_type', pd.Series.nunique),
        most_common_segment=('segment', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        most_common_beh_segment=('beh_segment', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        most_common_active_ind=('active_ind', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        tod_distribution=('tod', lambda x: x.value_counts(normalize=True).to_dict())
    ).reset_index()

    # Create behavioral ratios
    customer_data['click_to_checkout_ratio'] = customer_data['click_count'] / customer_data['checkout_count'].replace(0, np.nan)
    customer_data['click_to_checkout_ratio'] = customer_data['click_to_checkout_ratio'].fillna(0)

    # Expand time-of-day features
    print("Expanding time-of-day features...")
    for tod in ['early', 'morning', 'afternoon', 'evening']:
        customer_data[f'tod_{tod}'] = customer_data['tod_distribution'].apply(
            lambda d: d.get(tod.title(), 0))
    customer_data.drop(columns='tod_distribution', inplace=True)

    return customer_data

def detect_outliers(df):
    """Identify and remove outliers using Isolation Forest"""
    print("Detecting outliers...")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('idcol')
    
    iso_forest = IsolationForest(contamination=0.05, 
                                random_state=RANDOM_STATE)
    df['is_outlier'] = iso_forest.fit_predict(df[numeric_cols])
    return df[df['is_outlier'] == 1].drop(columns='is_outlier')

def plot_cumulative_variance(scaled_data, threshold=0.85):
    """
    Plots cumulative explained variance vs number of components
    Helps determine optimal PCA component count
    """
    print("Analyzing PCA variance...")
    # Fit PCA to all components
    pca_full = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find optimal component count
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold:.0%} Threshold')
    plt.axvline(x=optimal_components, color='g', linestyle='-', 
               label=f'Optimal: {optimal_components} Components')
    
    # Format plot
    plt.title('Cumulative Explained Variance by PCA Components', fontsize=16)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, len(cumulative_variance)+1))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Annotate current selection (3 components)
    current_variance = cumulative_variance[2] if len(cumulative_variance) > 2 else cumulative_variance[-1]
    plt.annotate(f'Current: {current_variance:.1%} with 3 PCs',
                xy=(3, current_variance),
                xytext=(3, current_variance - 0.1),
                arrowprops=dict(arrowstyle="->", color='purple'))
    
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png', dpi=300)
    plt.show()
    
    return optimal_components

def prepare_clustering_data(df):
    """Prepare data for clustering with scaling and PCA"""
    features = df.select_dtypes(include=np.number).columns.tolist()
    features.remove('idcol')
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Variance analysis
    optimal_components = plot_cumulative_variance(scaled_data)
    print(f"💡 Optimal components for 85% variance: {optimal_components}")
    
    # Apply PCA with 3 components (for visualization)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_data = pca.fit_transform(scaled_data)
    
    # Report variance captured
    explained_variance = sum(pca.explained_variance_ratio_)
    print(f"✅ Explained variance with 3 components: {explained_variance:.1%}")
    
    return pca_data, features, scaler, pca  # Return scaler and PCA for model saving

def optimized_elbow_analysis(data):
    """Determine optimal cluster count with visual and silhouette backup"""
    print("Finding optimal clusters...")
    visualizer = KElbowVisualizer(
        MiniBatchKMeans(init='k-means++', n_init=3, random_state=RANDOM_STATE),
        k=CLUSTER_RANGE,
        metric='distortion',
        timings=False,
        locate_elbow=False
    )
    visualizer.fit(data)
    visualizer.show()
    
    if visualizer.elbow_value_ is None:
        silhouette_scores = []
        for k in range(CLUSTER_RANGE[0], CLUSTER_RANGE[1] + 1):
            model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE)
            labels = model.fit_predict(data)
            silhouette_scores.append(silhouette_score(data, labels))
        return CLUSTER_RANGE[0] + np.argmax(silhouette_scores)
    return visualizer.elbow_value_

def perform_clustering(data, optimal_k):
    """Apply KMeans clustering with optimal cluster count"""
    print("Training KMeans model...")
    return KMeans(n_clusters=optimal_k, init='k-means++', 
                 n_init=10, max_iter=300, 
                 random_state=RANDOM_STATE).fit(data)

def plot_3d_clusters(df):
    """Create interactive 3D visualization of clusters in PCA space"""
    print("Generating 3D cluster visualization...")
    # Create separate DataFrames for each cluster
    clusters = [df[df['cluster'] == i] for i in sorted(df['cluster'].unique())]
    
    # Set professional color scheme (RGB)
    colors = ['#e8000b', '#1ac938', '#023eff', '#8b2be2', '#ff7c00', 
              '#00ccff', '#ff55a3', '#9c755f', '#5edc1f', '#720058'][:len(clusters)]
    
    fig = go.Figure()
    
    # Add traces for each cluster
    for i, cluster_df in enumerate(clusters):
        fig.add_trace(go.Scatter3d(
            x=cluster_df['PC1'], 
            y=cluster_df['PC2'], 
            z=cluster_df['PC3'],
            mode='markers',
            marker=dict(
                color=colors[i],
                size=5,
                opacity=0.4,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            name=f'Cluster {i}'
        ))
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(text='3D Visualization of Customer Clusters in PCA Space', 
                  x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
            yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
            zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
            camera=dict(up=dict(x=0, y=0, z=1))
        ),
        width=1000,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()

def plot_cluster_distribution(df):
    """Visualize percentage distribution across clusters"""
    print("Plotting cluster distribution...")
    # Calculate percentage distribution
    cluster_percentage = (df['cluster'].value_counts(normalize=True) * 100).reset_index()
    cluster_percentage.columns = ['Cluster', 'Percentage']
    cluster_percentage = cluster_percentage.sort_values('Cluster')
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Percentage', 
        y='Cluster', 
        data=cluster_percentage, 
        orient='h', 
        palette=COLORS,
        order=cluster_percentage['Cluster']
    )
    
    # Add percentage labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.5, 
            p.get_y() + p.get_height()/2, 
            f'{width:.1f}%', 
            ha='left', 
            va='center',
            fontsize=10
        )
    
    # Format plot
    plt.title('Customer Distribution Across Clusters', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.xticks(np.arange(0, 55, 5))
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=300)
    plt.show()

def print_cluster_metrics(X, clusters):
    """Calculate and print clustering metrics with interpretation"""
    print("Calculating cluster metrics...")
    # Calculate metrics
    metrics = [
        ["Number of Observations", len(X)],
        ["Silhouette Score", f"{silhouette_score(X, clusters):.3f}"],
        ["Calinski Harabasz Score", f"{calinski_harabasz_score(X, clusters):.1f}"],
        ["Davies Bouldin Score", f"{davies_bouldin_score(X, clusters):.3f}"]
    ]
    
    # Print table
    print("\n" + tabulate(metrics, headers=["Metric", "Value"], tablefmt="pretty"))
    
    # Add interpretation
    print("\nInterpretation:")
    print(f"- Silhouette Score ({float(metrics[1][1]):.3f}): ", end='')
    if float(metrics[1][1]) > 0.5:
        print("Reasonable structure (clusters are distinguishable)")
    elif float(metrics[1][1]) > 0.25:
        print("Weak structure (clusters may overlap)")
    else:
        print("No substantial structure (clusters not well separated)")
    
    print(f"- Calinski Harabasz ({float(metrics[2][1]):.1f}): ", end='')
    if float(metrics[2][1]) > 300:
        print("Well-defined clusters (good separation)")
    elif float(metrics[2][1]) > 150:
        print("Moderate separation")
    else:
        print("Poor separation (clusters may not be distinct)")
    
    print(f"- Davies Bouldin ({float(metrics[3][1]):.3f}): ", end='')
    if float(metrics[3][1]) < 0.5:
        print("Excellent separation")
    elif float(metrics[3][1]) < 0.8:
        print("Good separation")
    else:
        print("Overlapping clusters")

def plot_cluster_radar(df, n_clusters):
    """Create radar charts comparing standardized cluster centroids"""
    print("Generating radar charts...")
    # Prepare data (set idcol as index and keep only numeric features)
    df_temp = df.set_index('idcol')
    numeric_cols = df_temp.select_dtypes(include=np.number).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')  # Exclude cluster label
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_temp[numeric_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_cols, index=df_temp.index)
    df_scaled['cluster'] = df_temp['cluster']

    # Calculate cluster centroids
    centroids = df_scaled.groupby('cluster').mean()
    
    # Feature abbreviation mapping for cleaner labels
    feature_names = {
        'recency_days': 'Recency',
        'interaction_count': 'Interactions',
        'checkout_count': 'Checkouts',
        'click_count': 'Clicks',
        'click_to_checkout_ratio': 'C2C Ratio',
        'item_type_diversity': 'Item Diversity',
        'tod_early': 'Early',
        'tod_morning': 'Morning',
        'tod_afternoon': 'Afternoon',
        'tod_evening': 'Evening'
    }
    
    # Apply abbreviations
    features = [feature_names.get(col, col) for col in centroids.columns]
    
    # Prepare radar plot parameters
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete circle
    features += features[:1]  # For labeling
    
    # Create subplots
    n_cols = min(3, n_clusters)  # Max 3 columns per row
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), 
                          subplot_kw=dict(polar=True))
    
    # Handle single subplot case
    if n_clusters == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    
    # Generate radar plot for each cluster
    colors = sns.color_palette("Set2", n_colors=n_clusters)
    for i in range(n_clusters):
        ax = axs[i]
        # Get centroid values and complete circle
        values = centroids.loc[i].tolist()
        values += values[:1]
        
        # Plot data
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # Format plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features[:-1], fontsize=9)
        ax.set_title(f'Cluster {i}', size=14, color=colors[i], pad=20)
        ax.tick_params(pad=8)  # Add padding to labels
        ax.set_rlabel_position(30)  # Move radial labels
        ax.grid(color='grey', linewidth=0.5, alpha=0.5)
        
        # Add feature highlights
        max_val = max(values)
        for j, (angle, val) in enumerate(zip(angles[:-1], values[:-1])):
            if val > 1.0:  # Highlight strong positive characteristics
                ax.text(angle, val + 0.2, f"{val:.1f}", 
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=colors[i], alpha=0.7))
    
    # Hide unused subplots
    for j in range(n_clusters, len(axs)):
        axs[j].axis('off')
        
    plt.suptitle('Cluster Profile Comparison (Standardized Features)', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig('cluster_radar.png', dpi=300)
    plt.show()

def plot_feature_histograms_by_cluster(df, n_clusters):
    """
    Plot histograms of each feature segmented by cluster
    Shows distribution differences between clusters for each feature
    """
    print("Plotting feature histograms...")
    # Select numeric features excluding ID and cluster
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_cols if col not in ['idcol', 'cluster']]
    
    # Get sorted cluster list
    clusters = sorted(df['cluster'].unique())
    
    # Set up plot grid
    n_features = len(features)
    fig, axes = plt.subplots(n_features, n_clusters, 
                             figsize=(5*n_clusters, 3*n_features),
                             squeeze=False)
    
    # Use consistent cluster colors
    colors = sns.color_palette("Set2", n_colors=n_clusters)
    
    # Plot histograms for each feature-cluster combination
    for i, feature in enumerate(features):
        for j, cluster in enumerate(clusters):
            ax = axes[i, j]
            cluster_data = df[df['cluster'] == cluster][feature]
            
            # Plot histogram
            sns.histplot(cluster_data, ax=ax, color=colors[j], 
                         kde=True, bins=20, alpha=0.7, edgecolor='w')
            
            # Format plot
            ax.set_title(f'Cluster {cluster}', fontsize=12)
            if j == 0:  # Only add ylabel to first column
                ax.set_ylabel(feature, fontsize=10)
            if i == n_features - 1:  # Only add xlabel to last row
                ax.set_xlabel('Value', fontsize=9)
            else:
                ax.set_xlabel('')
                
            # Add vertical line at mean
            mean_val = cluster_data.mean()
            ax.axvline(mean_val, color='darkred', linestyle='dashed', linewidth=1.5)
            ax.annotate(f'μ={mean_val:.1f}', 
                        xy=(mean_val, ax.get_ylim()[1]*0.9),
                        xytext=(5, 0), textcoords='offset points',
                        color='darkred', fontsize=9)
    
    plt.suptitle('Feature Distributions by Cluster', fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    plt.savefig('feature_histograms.png', dpi=300)
    plt.show()

def profile_clusters(clean_data, transaction_df, optimal_k):
    """
    Comprehensive cluster profiling with original transaction data
    """
    print("Profiling clusters...")
    # Merge cluster labels with original transactions
    merged_df = transaction_df.merge(
        clean_data[['idcol', 'cluster']],
        on='idcol',
        how='inner'
    )
    
    print("\n🔍 Deep Cluster Profiling:")
    
    # 1. Item Type Analysis
    print("\n📦 Item Type Distribution per Cluster:")
    item_type_profile = merged_df.groupby('cluster')['item_type'].value_counts(normalize=True).unstack()
    item_type_profile = item_type_profile.fillna(0)
    
    # Format percentages
    def format_percent(x):
        return f"{x:.1%}"
    
    formatted_item = item_type_profile.applymap(format_percent)
    print(tabulate(formatted_item, headers='keys', tablefmt="pretty"))
    
    # 2. Item Description Analysis
    print("\n🏷️ Top Item Descriptions per Cluster:")
    top_items = merged_df.groupby(['cluster', 'item_descrip']).size().reset_index(name='count')
    top_items = top_items.sort_values(['cluster', 'count'], ascending=[True, False])
    top_items = top_items.groupby('cluster').head(5)
    
    for cluster in range(optimal_k):
        cluster_items = top_items[top_items['cluster'] == cluster]
        print(f"\nCluster {cluster} Top Items:")
        print(tabulate(cluster_items[['item_descrip', 'count']], 
                      headers=['Item Description', 'Count'],
                      tablefmt="pretty"))
    
    # 3. Time-of-Day Engagement
    print("\n⏰ Time-of-Day Engagement Patterns:")
    # Get actual time-of-day distribution from transactions
    tod_profile = merged_df.groupby(['cluster', 'tod']).size().reset_index(name='count')
    tod_pivot = tod_profile.pivot(index='tod', columns='cluster', values='count').fillna(0)
    
    # Convert counts to percentages
    tod_percent = tod_pivot.div(tod_pivot.sum(axis=0), axis=1).applymap(format_percent)
    print(tabulate(tod_percent, headers='keys', tablefmt="pretty"))
    
    # 4. Visualization: Item Type Distribution
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        data=item_type_profile.reset_index().melt(id_vars='cluster'),
        x='value',
        y='item_type',
        hue='cluster',
        kind='bar',
        col='cluster',
        col_wrap=min(3, optimal_k),
        height=6,
        aspect=1.2,
        palette=COLORS[:optimal_k],
        sharex=False
    )
    g.set_axis_labels("Proportion", "Item Type")
    g.set_titles("Cluster {col_name}")
    g.fig.suptitle('Item Type Distribution by Cluster', y=1.03, fontsize=18)
    plt.tight_layout()
    plt.savefig('item_type_distribution.png', dpi=300)
    plt.show()
    
    # 5. Visualization: Time-of-Day Patterns
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=tod_profile,
        x='tod',
        y='count',
        hue='cluster',
        palette=COLORS[:optimal_k],
        errorbar=None
    )
    plt.title('Time-of-Day Engagement by Cluster', fontsize=18)
    plt.xlabel('Time of Day')
    plt.ylabel('Interaction Count')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    total_counts = tod_profile.groupby('cluster')['count'].transform('sum')
    tod_profile['percentage'] = tod_profile['count'] / total_counts * 100
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50,
                f'{height:.0f}',
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig('time_of_day.png', dpi=300)
    plt.show()
    
    # 6. Behavioral Segment Heatmap
    beh_segment_profile = clean_data.groupby(['cluster', 'most_common_beh_segment']).size().unstack().fillna(0)
    plt.figure(figsize=(16, 10))
    sns.heatmap(beh_segment_profile, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Behavioral Segment Distribution Across Clusters', fontsize=18)
    plt.xlabel('Behavioral Segment')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('behavioral_segments.png', dpi=300)
    plt.show()

def analyze_segment_item_affinity(merged_df):
    """
    Analyzes item preferences for each segment within clusters
    """
    print("Analyzing segment-item affinity...")
    # 1. Segment-Item Type Analysis
    print("\n📊 Item Type Preferences by Segment and Cluster:")
    segment_item_type = merged_df.groupby(
        ['cluster', 'most_common_segment', 'item_type']
    ).size().reset_index(name='count')
    
    # Find dominant item types per segment-cluster
    dominant_types = segment_item_type.loc[
        segment_item_type.groupby(['cluster', 'most_common_segment'])['count'].idxmax()
    ]
    print("\nDominant Item Types per Segment-Cluster:")
    print(tabulate(dominant_types, headers='keys', tablefmt="pretty"))
    
    # 2. Segment-Item Description Analysis
    print("\n🏷️ Top Items by Segment and Cluster:")
    segment_items = merged_df.groupby(
        ['cluster', 'most_common_segment', 'item_descrip']
    ).size().reset_index(name='count')
    
    # Get top 3 items per segment-cluster
    top_segment_items = segment_items.sort_values(
        ['cluster', 'most_common_segment', 'count'], 
        ascending=[True, True, False]
    ).groupby(['cluster', 'most_common_segment']).head(3)
    
    # Pivot for better readability
    pivot_items = top_segment_items.pivot_table(
        index=['cluster', 'most_common_segment'],
        columns='item_descrip',
        values='count',
        fill_value=0
    )
    print(tabulate(pivot_items, headers='keys', tablefmt="pretty"))
    
    # 3. Visualization: Segment-Item Heatmap
    plt.figure(figsize=(16, 12))
    # Fixed crosstab implementation
    cross_tab = pd.crosstab(
        index=merged_df['cluster'], 
        columns=merged_df['most_common_segment'], 
        values=merged_df['item_descrip'],
        aggfunc='count',
        normalize='index'
    )
    sns.heatmap(cross_tab, annot=True, fmt='.1%', cmap='YlGnBu')
    plt.title('Item Purchase Distribution by Cluster and Segment', fontsize=18)
    plt.xlabel('Customer Segment')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('segment_item_heatmap.png', dpi=300)
    plt.show()
    
    # 4. Visualization: Segment-Item Type Network
    plt.figure(figsize=(16, 12))
    segment_item_counts = merged_df.groupby(
        ['most_common_segment', 'item_type']
    ).size().reset_index(name='count')
    
    # Create bubble chart
    plt.scatter(
        x=segment_item_counts['most_common_segment'],
        y=segment_item_counts['item_type'],
        s=segment_item_counts['count']/100,
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add labels
    for i, row in segment_item_counts.iterrows():
        if row['count'] > 1000:  # Only label significant connections
            plt.text(
                row['most_common_segment'], 
                row['item_type'], 
                f"{row['count']}",
                fontsize=9,
                ha='center'
            )
    
    plt.title('Segment-Item Type Affinity Network', fontsize=18)
    plt.xlabel('Customer Segment')
    plt.ylabel('Item Type')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('affinity_network.png', dpi=300)
    plt.show()

def generate_recommendations(clean_data, transaction_df):
    """
    Optimized product recommendation generation using vectorized operations
    with fixed item descriptions and fallback recommendations
    """
    print(f"Generating recommendations for {len(clean_data)} customers...")
    print(f"Current memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")
    
    # Create reliable item-description mapping
    item_desc_map = (
        transaction_df[['item', 'item_descrip']]
        .drop_duplicates(subset=['item'])
        .set_index('item')['item_descrip']
        .to_dict()
    )
    
    # Optimize data types
    transaction_df = transaction_df.copy()
    clean_data = clean_data.copy()
    clean_data['cluster'] = clean_data['cluster'].astype('int8')
    
    # Step 1: Merge cluster information with transactions
    merged_data = transaction_df.merge(
        clean_data[['idcol', 'cluster']],
        on='idcol',
        how='inner'
    )
    
    # Step 2: Filter to only checkout interactions
    checkout_data = merged_data[merged_data['interaction'] == 'checkout']
    
    # Step 3: Get best-selling products per cluster
    best_selling = (
        checkout_data
        .groupby(['cluster', 'item'])
        .size()
        .reset_index(name='purchase_count')
        .sort_values(['cluster', 'purchase_count'], ascending=[True, False])
    )
    
    # Add descriptions
    best_selling['item_descrip'] = best_selling['item'].map(item_desc_map)
    
    # Get top 20 products per cluster (expanded pool)
    top_products = (
        best_selling
        .groupby('cluster')
        .head(20)
        .reset_index(drop=True)
    )
    
    # Get global top products for fallback
    global_top = best_selling.head(10)
    
    # Step 4: Create customer purchase records
    customer_purchases = (
        checkout_data
        .groupby(['idcol', 'item'])
        .size()
        .reset_index(name='purchased')
        [['idcol', 'item']]
    )
    
    # Step 5: Vectorized recommendation generation
    # Create cross-join between customers and top cluster products
    customer_clusters = clean_data[['idcol', 'cluster']]
    recommendations_base = (
        customer_clusters
        .merge(top_products, on='cluster', how='left')
    )
    
    # Flag already purchased items
    recommendations_base = (
        recommendations_base
        .merge(
            customer_purchases,
            on=['idcol', 'item'],
            how='left',
            indicator='purchased_flag'
        )
    )
    recommendations_base['purchased'] = recommendations_base['purchased_flag'] == 'both'
    
    # Filter to unpurchased items
    new_recommendations = recommendations_base[~recommendations_base['purchased']]
    
    # Rank recommendations by purchase_count within customer
    new_recommendations['rank'] = (
        new_recommendations
        .groupby('idcol')['purchase_count']
        .rank(method='first', ascending=False)
    )
    
    # Select top 3 recommendations per customer
    top_recs = (
        new_recommendations[new_recommendations['rank'] <= 3]
        .sort_values(['idcol', 'rank'])
    )
    
    # Pivot to wide format
    recs_pivoted = (
        top_recs
        .assign(rec_num=lambda x: 'rec' + x['rank'].astype(int).astype(str))
        .pivot_table(
            index=['idcol', 'cluster'],
            columns='rec_num',
            values=['item', 'item_descrip'],
            aggfunc='first'
        )
    )
    
    # Flatten multi-index columns
    recs_pivoted.columns = [
        f"{col[1]}_{col[0]}" for col in recs_pivoted.columns
    ]
    recs_pivoted = recs_pivoted.reset_index()
    
    # Step 6: Create recommendations dataframe
    rec_columns = [
        'idcol', 'cluster',
        'rec1_item', 'rec1_desc',
        'rec2_item', 'rec2_desc',
        'rec3_item', 'rec3_desc'
    ]
    
    # Ensure all columns exist
    for col in rec_columns:
        if col not in recs_pivoted.columns:
            recs_pivoted[col] = None

    print("Applying item descriptions...")
    for rec_num in range(1, 4):
        item_col = f'rec{rec_num}_item'
        desc_col = f'rec{rec_num}_desc'
        
        if item_col in recs_pivoted.columns:
            recs_pivoted[desc_col] = recs_pivoted[item_col].map(item_desc_map)
    
    # Step 7: Handle customers with no recommendations
    # Identify customers missing recommendations
    no_rec_customers = clean_data[
        ~clean_data['idcol'].isin(recs_pivoted['idcol'])
    ][['idcol', 'cluster']]
    
    if not no_rec_customers.empty:
        # Assign global top products to these customers
        global_recs = []
        for customer in no_rec_customers.itertuples():
            rec_list = []
            for i, row in global_top.head(3).iterrows():
                rec_list.extend([row['item'], row['item_descrip']])
            # Pad if needed
            while len(rec_list) < 6:
                rec_list.extend([None, None])
            global_recs.append([customer.idcol, customer.cluster] + rec_list)
        
        global_recs_df = pd.DataFrame(global_recs, columns=rec_columns)
        recs_pivoted = pd.concat([recs_pivoted, global_recs_df], ignore_index=True)
    
    return recs_pivoted[rec_columns]

def plot_feature_importance(pca, features):
    """Visualize PCA feature loadings"""
    # Create loading matrix
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=features
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings.abs(), annot=True, cmap='Blues', fmt=".2f", 
                linewidths=0.5, cbar_kws={'label': 'Loading Strength'})
    plt.title('Feature Contributions to Principal Components', fontsize=16)
    plt.ylabel('Original Features')
    plt.xlabel('Principal Components')
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=300)
    plt.show()

def main():
    """Main analysis pipeline"""
    print("🔄 Loading and preprocessing data...")
    raw_data = load_and_preprocess()
    
    print("\n🔍 Detecting outliers...")
    clean_data = detect_outliers(raw_data)
    
    print("\n⚙️ Preparing clustering data...")
    # Extract features and scale
    pca_data, features, scaler, pca = prepare_clustering_data(clean_data)
    customer_data_pca = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(3)])
    customer_data_pca['idcol'] = clean_data['idcol'].values
    
    print(f"✅ Explained variance with 3 components: {sum(pca.explained_variance_ratio_):.1%}")
    
    # Feature importance visualization
    print("\n📊 Visualizing feature contributions to PCA...")
    plot_feature_importance(pca, features)
    
    # Sample data for elbow analysis
    sample_idx = np.random.choice(pca_data.shape[0], int(pca_data.shape[0] * SAMPLE_FRAC), replace=False)
    sample_data = pca_data[sample_idx]
    
    print("\n📊 Determining optimal clusters...")
    optimal_k = optimized_elbow_analysis(sample_data)
    optimal_k = min(max(optimal_k, CLUSTER_RANGE[0]), CLUSTER_RANGE[1])
    print(f"✅ Optimal cluster count: {optimal_k}")
    
    print("\n🎯 Training final model...")
    final_model = perform_clustering(pca_data, optimal_k)
    clean_data['cluster'] = final_model.labels_
    customer_data_pca['cluster'] = final_model.labels_
    
    # Save results
    clean_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Results saved to {OUTPUT_PATH}")
    
    # Save full model pipeline
    model_pipeline = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': final_model,
        'features': features
    }
    model_path = OUTPUT_PATH.replace('.csv', '_model.joblib')
    dump(model_pipeline, model_path)
    print(f"💾 Full model pipeline saved to {model_path}")
    
    # Cluster evaluation
    print("\n📈 Calculating cluster metrics...")
    X = customer_data_pca.drop(['cluster', 'idcol'], axis=1)
    clusters = customer_data_pca['cluster']
    print_cluster_metrics(X, clusters)
    
    # Visualizations
    print("\n📊 Generating cluster visualizations...")
   # plot_3d_clusters(customer_data_pca)
   # plot_cluster_distribution(clean_data)
   # plot_cluster_radar(clean_data, optimal_k)
    
    # Add feature histograms
    print("\n📊 Plotting feature distributions by cluster...")
    #plot_feature_histograms_by_cluster(clean_data, optimal_k)

    print("\n🔍 Loading original transaction data for profiling...")
    transaction_df = pd.read_csv(INPUT_PATH, encoding='utf-8', sep=';')
    
    # Enhanced cluster profiling
    print("\n📊 Profiling clusters with transaction data...")
    profile_clusters(clean_data, transaction_df, optimal_k)
    
    # Segment-item affinity analysis
    print("\n🔗 Analyzing segment-item affinities...")
    merged_df = transaction_df.merge(
        clean_data[['idcol', 'cluster', 'most_common_segment']],
        on='idcol'
    )
    analyze_segment_item_affinity(merged_df)

    # Generate recommendations in chunks
    print("\n🎯 Generating personalized recommendations...")
    if len(clean_data) > RECOMMENDATION_CHUNK_SIZE:
        recommendations = []
        for i in tqdm(range(0, len(clean_data), RECOMMENDATION_CHUNK_SIZE),
                          desc="Processing customer chunks"):
            chunk = clean_data.iloc[i:i+RECOMMENDATION_CHUNK_SIZE]
            rec_df = generate_recommendations(chunk, transaction_df)
            recommendations.append(rec_df)
        rec_df = pd.concat(recommendations)
    else:
        rec_df = generate_recommendations(clean_data, transaction_df)
    
    # Merge with customer data
    customer_recs = clean_data.merge(rec_df, on=['idcol', 'cluster'], how='left')
    
    # Save recommendations
    rec_output = OUTPUT_PATH.replace('.csv', '_recommendations.csv')
    customer_recs.to_csv(rec_output, index=False)
    print(f"\n💾 Recommendations saved to {rec_output}")
    
    # Show sample recommendations
    print("\n✨ Sample Recommendations:")
    sample_cols = ['idcol', 'cluster', 'rec1_item', 'rec1_desc',
                   'rec2_item', 'rec2_desc', 'rec3_item', 'rec3_desc']
    
    sample_recs = customer_recs[sample_cols].sample(5)
    print(tabulate(sample_recs.fillna(''), headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    main()