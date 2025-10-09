import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# Step 1: Prepare core data
print("Step 1: Preparing data...")
df = new_scores_with_names.merge(label_names[['PIN', 'label']], on='PIN', how='inner')
emb_cols = [col for col in df.columns if col.startswith('emb_')]
embeddings = df[emb_cols].values

print(f"Data prepared: {len(df)} providers, {len(emb_cols)} embedding dimensions")
print(f"Unique labels: {df['label'].nunique()}")
print(f"Labels: {sorted(df['label'].unique())}")

# Step 2: Calculate centroids for each label
print("\nStep 2: Calculating centroids...")
labels = df['label'].unique()
centroids = {}

for label in labels:
    mask = df['label'] == label
    count = mask.sum()
    centroids[label] = embeddings[mask].mean(axis=0)
    print(f"  {label}: {count} providers")

print(f"Centroids calculated for {len(centroids)} labels")

# Step 3: Calculate distances from each provider to all centroids
print("\nStep 3: Calculating distances to centroids...")
distance_data = []

for label, centroid in centroids.items():
    distances = cosine_distances(embeddings, centroid.reshape(1, -1)).flatten()
    distance_data.append(pd.DataFrame({
        'PIN': df['PIN'].values,
        'distance': distances,
        'target_label': label
    }))

distances_df = pd.concat(distance_data, ignore_index=True)
distances_df = distances_df.merge(df[['PIN', 'label']], on='PIN')

print(f"Distance matrix created: {len(distances_df)} rows")
print(f"Sample distances:")
sample = distances_df.head()
print(sample)

# Step 4: Calculate median distances for each label to its own centroid
print("\nStep 4: Calculating group medians...")
own_distances = distances_df[distances_df['label'] == distances_df['target_label']]
medians = own_distances.groupby('label')['distance'].median().to_dict()

print("Median distances to own centroid:")
for label, median_dist in medians.items():
    print(f"  {label}: {median_dist:.4f}")

# Step 5: Find outliers within each group
print("\nStep 5: Finding outliers...")
own_distances['median_distance'] = own_distances['label'].map(medians)
outliers = own_distances[own_distances['distance'] > own_distances['median_distance']]

print(f"Outliers found: {len(outliers)} providers")
print("Outliers by label:")
outlier_counts = outliers['label'].value_counts()
print(outlier_counts)

# Step 6: Check cross-group distances for outliers
print("\nStep 6: Detecting overlaps...")
overlaps = []

for _, outlier in outliers.iterrows():
    pin = outlier['PIN']
    own_label = outlier['label']
    own_distance = outlier['distance']
    
    # Get distances to other centroids
    other_distances = distances_df[
        (distances_df['PIN'] == pin) & 
        (distances_df['target_label'] != own_label)
    ]
    
    for _, other in other_distances.iterrows():
        other_label = other['target_label']
        other_distance = other['distance']
        other_median = medians[other_label]
        
        # Overlap condition: closer to other centroid than its median
        if other_distance < other_median:
            overlap_score = own_distance / other_distance
            overlaps.append({
                'outlier_pin': pin,
                'outlier_label': own_label,
                'overlap_label': other_label,
                'own_distance': own_distance,
                'other_distance': other_distance,
                'own_median': medians[own_label],
                'other_median': other_median,
                'overlap_score': overlap_score
            })

overlaps_df = pd.DataFrame(overlaps)

print(f"Overlaps detected: {len(overlaps_df)}")

if len(overlaps_df) > 0:
    print("\nTop overlaps:")
    top_overlaps = overlaps_df.nlargest(10, 'overlap_score')
    
    # Add provider names for readability
    top_overlaps = top_overlaps.merge(
        new_scores_with_names[['PIN', 'PIN_name']], 
        left_on='outlier_pin', right_on='PIN', how='left'
    ).drop('PIN', axis=1)
    
    display_cols = ['PIN_name', 'outlier_label', 'overlap_label', 
                   'own_distance', 'other_distance', 'overlap_score']
    print(top_overlaps[display_cols])
else:
    print("No overlaps detected with current criteria")

# Step 7: Summary statistics
print(f"\nStep 7: Summary statistics...")
if len(overlaps_df) > 0:
    overlap_summary = overlaps_df.groupby(['outlier_label', 'overlap_label']).agg({
        'outlier_pin': 'count',
        'overlap_score': ['mean', 'max']
    }).round(3)
    overlap_summary.columns = ['overlap_count', 'avg_score', 'max_score']
    overlap_summary = overlap_summary.reset_index()
    
    print("Overlap summary by label pairs:")
    print(overlap_summary.sort_values('overlap_count', ascending=False))
    
    print(f"\nTotal overlap cases: {len(overlaps_df)}")
    print(f"Unique label pairs with overlaps: {len(overlap_summary)}")
    print(f"Labels with outliers: {overlaps_df['outlier_label'].nunique()}")
else:
    print("No overlap summary available")

print("\nStage 0 complete!")