from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

print("Stage 1: Analysis Preparation")
print("=" * 40)

# Step 1: Find typical providers (closest to each centroid)
print("Step 1: Finding typical providers...")
own_distances = distances_df[distances_df['label'] == distances_df['target_label']]
typical_providers = own_distances.loc[own_distances.groupby('label')['distance'].idxmin()]
typical_lookup = typical_providers[['label', 'PIN']].set_index('label')['PIN'].to_dict()

print("Typical providers (closest to centroid):")
for label, pin in typical_lookup.items():
    name = new_scores_with_names[new_scores_with_names['PIN'] == pin]['PIN_name'].iloc[0]
    distance = typical_providers[typical_providers['PIN'] == pin]['distance'].iloc[0]
    print(f"  {label}: {name} (distance: {distance:.4f})")

# Step 2: Enhance overlaps with provider names and typical providers
print("\nStep 2: Enhancing overlap data...")
enhanced_overlaps = overlaps_df.copy()

# Add outlier provider name
enhanced_overlaps = enhanced_overlaps.merge(
    new_scores_with_names[['PIN', 'PIN_name']], 
    left_on='outlier_pin', right_on='PIN', how='left'
).drop('PIN', axis=1).rename(columns={'PIN_name': 'outlier_name'})

# Add typical providers for comparison
enhanced_overlaps['typical_same_pin'] = enhanced_overlaps['outlier_label'].map(typical_lookup)
enhanced_overlaps['typical_other_pin'] = enhanced_overlaps['overlap_label'].map(typical_lookup)

# Add typical provider names
typical_same_names = enhanced_overlaps['typical_same_pin'].map(
    new_scores_with_names.set_index('PIN')['PIN_name'].to_dict()
)
typical_other_names = enhanced_overlaps['typical_other_pin'].map(
    new_scores_with_names.set_index('PIN')['PIN_name'].to_dict()
)

enhanced_overlaps['typical_same_name'] = typical_same_names
enhanced_overlaps['typical_other_name'] = typical_other_names

print(f"Enhanced overlaps: {len(enhanced_overlaps)} cases")

# Step 3: Create analysis combinations table
print("\nStep 3: Creating analysis combinations...")
analysis_combinations = []

for _, row in enhanced_overlaps.iterrows():
    combo = {
        'analysis_id': f"{row['outlier_label']}_vs_{row['overlap_label']}_{row['outlier_pin']}",
        'analysis_description': f"{row['outlier_label']} vs {row['overlap_label']} - {row['outlier_name']}",
        'outlier_pin': row['outlier_pin'],
        'outlier_name': row['outlier_name'],
        'outlier_label': row['outlier_label'],
        'overlap_label': row['overlap_label'],
        'typical_same_pin': row['typical_same_pin'],
        'typical_same_name': row['typical_same_name'],
        'typical_other_pin': row['typical_other_pin'],
        'typical_other_name': row['typical_other_name'],
        'own_distance': row['own_distance'],
        'other_distance': row['other_distance'],
        'overlap_score': row['overlap_score'],
        'comparison_summary': f"Outlier: {row['outlier_name']} vs Same: {row['typical_same_name']} vs Other: {row['typical_other_name']}"
    }
    analysis_combinations.append(combo)

analysis_df = pd.DataFrame(analysis_combinations)

print(f"Analysis combinations created: {len(analysis_df)}")
print("\nSample combinations:")
print(analysis_df[['analysis_description', 'comparison_summary']].head())

# Step 4: Generate t-SNE coordinates
print("\nStep 4: Generating t-SNE coordinates...")
print("Computing t-SNE (this may take a moment)...")

perplexity = min(30, len(embeddings) // 4)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
tsne_coords = tsne.fit_transform(embeddings)

tsne_df = pd.DataFrame({
    'PIN': df['PIN'].values,
    'tsne_x': tsne_coords[:, 0],
    'tsne_y': tsne_coords[:, 1],
    'label': df['label'].values
})

# Add provider names to t-SNE data
tsne_df = tsne_df.merge(new_scores_with_names[['PIN', 'PIN_name']], on='PIN', how='left')

print(f"t-SNE coordinates generated for {len(tsne_df)} providers")

# Step 5: Create summary statistics
print("\nStep 5: Creating summary statistics...")

# Overlap summary by label pairs
overlap_summary = enhanced_overlaps.groupby(['outlier_label', 'overlap_label']).agg({
    'outlier_pin': 'count',
    'overlap_score': ['mean', 'max', 'min'],
    'own_distance': 'mean',
    'other_distance': 'mean'
}).round(4)

overlap_summary.columns = ['count', 'avg_score', 'max_score', 'min_score', 'avg_own_dist', 'avg_other_dist']
overlap_summary = overlap_summary.reset_index().sort_values('count', ascending=False)

print("Overlap summary by label pairs:")
print(overlap_summary)

# Provider-level summary
provider_summary = enhanced_overlaps.groupby(['outlier_pin', 'outlier_name', 'outlier_label']).agg({
    'overlap_label': 'count',
    'overlap_score': 'max'
}).rename(columns={'overlap_label': 'overlap_count', 'overlap_score': 'max_overlap_score'})
provider_summary = provider_summary.reset_index().sort_values('overlap_count', ascending=False)

print(f"\nTop providers with most overlaps:")
print(provider_summary.head())

# Step 6: Data validation
print("\nStep 6: Data validation...")
print(f"Total providers in dataset: {len(df)}")
print(f"Providers with overlaps: {enhanced_overlaps['outlier_pin'].nunique()}")
print(f"Unique analysis combinations: {len(analysis_df)}")
print(f"Label pairs with overlaps: {len(overlap_summary)}")

# Check for any missing data
missing_checks = {
    'Missing typical same providers': enhanced_overlaps['typical_same_pin'].isna().sum(),
    'Missing typical other providers': enhanced_overlaps['typical_other_pin'].isna().sum(),
    'Missing outlier names': enhanced_overlaps['outlier_name'].isna().sum(),
    'Missing t-SNE coordinates': tsne_df[['tsne_x', 'tsne_y']].isna().sum().sum()
}

for check, count in missing_checks.items():
    if count > 0:
        print(f"WARNING: {check}: {count}")
    else:
        print(f"✓ {check}: None")

print("\nStage 1 complete! Data ready for interactive analysis.")
print("\nGenerated objects:")
print("- enhanced_overlaps: Enhanced overlap data with names and typical providers")
print("- analysis_df: Ready-to-use analysis combinations")
print("- tsne_df: t-SNE coordinates for visualization")
print("- overlap_summary: Summary statistics by label pairs")
print("- typical_lookup: Dictionary of typical providers by label")