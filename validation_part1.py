import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Filter to only the 7,990 labeled providers
labeled_pins = list(pin_to_label.keys())

print(f"Total providers in data: {len(pin_list)}")
print(f"Labeled providers: {len(labeled_pins)}")

# Find indices of labeled pins in pin_list (alignment already guaranteed)
pin_to_idx = {pin: idx for idx, pin in enumerate(pin_list)}
labeled_indices = [pin_to_idx[pin] for pin in labeled_pins if pin in pin_to_idx]

print(f"Matched indices: {len(labeled_indices)}")

# Filter embedding_df to only labeled providers
embedding_df_filtered = embedding_df[embedding_df['PIN'].isin(labeled_pins)].reset_index(drop=True)

# Filter matrices to only labeled providers (rows) - alignment preserved
diag_matrix_filtered = diag_matrix[labeled_indices, :]
proc_matrix_filtered = proc_matrix[labeled_indices, :]

# Create filtered pin_list - alignment preserved
filtered_pin_list = [pin_list[i] for i in labeled_indices]

print(f"\nFiltered shapes:")
print(f"embedding_df: {embedding_df_filtered.shape}")
print(f"diag_matrix: {diag_matrix_filtered.shape}")
print(f"proc_matrix: {proc_matrix_filtered.shape}")
print(f"filtered_pin_list length: {len(filtered_pin_list)}")

# Verify we have all labeled providers
print(f"\nAll labeled PINs found: {len(labeled_indices) == len(labeled_pins)}")


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate 1M random pairs from 7,990 providers
n_providers = len(filtered_pin_list)
n_pairs = 1_000_000

print(f"Generating {n_pairs:,} random pairs from {n_providers} providers...")

# Generate random pairs (avoid self-pairs)
idx1 = np.random.randint(0, n_providers, n_pairs)
idx2 = np.random.randint(0, n_providers, n_pairs)

# Remove self-pairs
mask = idx1 != idx2
idx1 = idx1[mask]
idx2 = idx2[mask]

# If we lost some pairs due to self-pairs, generate more
while len(idx1) < n_pairs:
    additional = n_pairs - len(idx1)
    new_idx1 = np.random.randint(0, n_providers, additional)
    new_idx2 = np.random.randint(0, n_providers, additional)
    new_mask = new_idx1 != new_idx2
    idx1 = np.concatenate([idx1, new_idx1[new_mask]])
    idx2 = np.concatenate([idx2, new_idx2[new_mask]])

# Take exactly 1M pairs
idx1 = idx1[:n_pairs]
idx2 = idx2[:n_pairs]

print(f"Generated {len(idx1):,} pairs")

# Step 2: Calculate procedure overlaps (cosine similarity of raw procedure vectors)
print("\nCalculating procedure overlaps from raw data...")

# We'll calculate in batches to avoid memory issues
batch_size = 10_000
n_batches = n_pairs // batch_size

proc_overlaps = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    
    batch_idx1 = idx1[start_idx:end_idx]
    batch_idx2 = idx2[start_idx:end_idx]
    
    # Get procedure vectors for this batch
    proc_vec1 = proc_matrix_filtered[batch_idx1]
    proc_vec2 = proc_matrix_filtered[batch_idx2]
    
    # Calculate cosine similarity for each pair
    batch_overlaps = np.array([
        cosine_similarity(proc_vec1[j:j+1], proc_vec2[j:j+1])[0, 0]
        for j in range(len(batch_idx1))
    ])
    
    proc_overlaps.extend(batch_overlaps)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1)*batch_size:,} pairs...")

proc_overlaps = np.array(proc_overlaps)
print(f"Procedure overlaps calculated: {len(proc_overlaps):,}")

# Step 3: Extract Tower 1 embeddings (procedures)
print("\nExtracting Tower 1 embeddings...")

tower1_cols = [col for col in embedding_df_filtered.columns if col.startswith('tower1_proc_emb_')]
print(f"Tower 1 columns: {len(tower1_cols)}")

tower1_embeddings = embedding_df_filtered[tower1_cols].values

# Step 4: Calculate embedding similarities for the same pairs
print("\nCalculating Tower 1 embedding similarities...")

emb_similarities = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    
    batch_idx1 = idx1[start_idx:end_idx]
    batch_idx2 = idx2[start_idx:end_idx]
    
    # Get embeddings for this batch
    emb_vec1 = tower1_embeddings[batch_idx1]
    emb_vec2 = tower1_embeddings[batch_idx2]
    
    # Calculate cosine similarity for each pair
    batch_sims = np.array([
        np.dot(emb_vec1[j], emb_vec2[j]) / (np.linalg.norm(emb_vec1[j]) * np.linalg.norm(emb_vec2[j]))
        for j in range(len(batch_idx1))
    ])
    
    emb_similarities.extend(batch_sims)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1)*batch_size:,} pairs...")

emb_similarities = np.array(emb_similarities)
print(f"Embedding similarities calculated: {len(emb_similarities):,}")

# Step 5: Calculate correlations
print("\n" + "="*60)
print("RESULTS: Procedure Overlap vs Tower 1 Embedding Similarity")
print("="*60)

pearson_r, pearson_p = pearsonr(proc_overlaps, emb_similarities)
spearman_r, spearman_p = spearmanr(proc_overlaps, emb_similarities)

print(f"\nPearson correlation:  r = {pearson_r:.4f} (p = {pearson_p:.2e})")
print(f"Spearman correlation: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

print(f"\nProcedure overlap stats:")
print(f"  Mean: {proc_overlaps.mean():.4f}")
print(f"  Std:  {proc_overlaps.std():.4f}")
print(f"  Min:  {proc_overlaps.min():.4f}")
print(f"  Max:  {proc_overlaps.max():.4f}")

print(f"\nEmbedding similarity stats:")
print(f"  Mean: {emb_similarities.mean():.4f}")
print(f"  Std:  {emb_similarities.std():.4f}")
print(f"  Min:  {emb_similarities.min():.4f}")
print(f"  Max:  {emb_similarities.max():.4f}")

# Step 6: Create visualization
print("\nCreating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot with hexbin for density
ax1 = axes[0]
hb = ax1.hexbin(proc_overlaps, emb_similarities, gridsize=50, cmap='YlOrRd', mincnt=1)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
ax1.set_xlabel('Procedure Overlap (Raw Data Cosine Similarity)', fontsize=12)
ax1.set_ylabel('Tower 1 Embedding Similarity', fontsize=12)
ax1.set_title(f'Procedure Overlap vs Embedding Similarity\nPearson r = {pearson_r:.4f}', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(hb, ax=ax1, label='Number of pairs')

# 2D histogram
ax2 = axes[1]
h = ax2.hist2d(proc_overlaps, emb_similarities, bins=50, cmap='YlOrRd', cmin=1)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
ax2.set_xlabel('Procedure Overlap (Raw Data Cosine Similarity)', fontsize=12)
ax2.set_ylabel('Tower 1 Embedding Similarity', fontsize=12)
ax2.set_title(f'2D Histogram\nSpearman ρ = {spearman_r:.4f}', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(h[3], ax=ax2, label='Number of pairs')

plt.tight_layout()
plt.savefig('step1_procedure_overlap_validation.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'step1_procedure_overlap_validation.png'")

plt.show()

print("\n" + "="*60)
print("Step 1 Complete!")
print("="*60)



import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Filter proc_grouped to only labeled providers
print("Filtering proc_grouped to labeled providers...")
print(f"Original proc_grouped shape: {proc_grouped.shape}")

proc_grouped_filtered = proc_grouped[proc_grouped['PIN'].isin(filtered_pin_list)].copy()

print(f"Filtered proc_grouped shape: {proc_grouped_filtered.shape}")
print(f"Unique PINs in filtered data: {proc_grouped_filtered['PIN'].nunique()}")

# Step 2: Prepare procedure lookup for labeled providers
print("\nPreparing procedure lookup...")

# Create a dictionary: PIN -> set of procedure codes
pin_to_procedures = defaultdict(set)
for _, row in proc_grouped_filtered.iterrows():
    pin_to_procedures[row['PIN']].add(row['code'])

print(f"Procedure data loaded for {len(pin_to_procedures)} providers")

# Verify all labeled providers have procedure data
missing_pins = set(filtered_pin_list) - set(pin_to_procedures.keys())
if missing_pins:
    print(f"WARNING: {len(missing_pins)} labeled providers have no procedure data")
else:
    print("✓ All labeled providers have procedure data")

# Step 3: Generate 10,000 random pairs
n_providers = len(filtered_pin_list)
n_pairs = 10_000

print(f"\nGenerating {n_pairs:,} random pairs from {n_providers} providers...")

# Generate random pairs (avoid self-pairs)
idx1 = np.random.randint(0, n_providers, n_pairs)
idx2 = np.random.randint(0, n_providers, n_pairs)

# Remove self-pairs
mask = idx1 != idx2
idx1 = idx1[mask]
idx2 = idx2[mask]

# If we lost some pairs, generate more
while len(idx1) < n_pairs:
    additional = n_pairs - len(idx1)
    new_idx1 = np.random.randint(0, n_providers, additional)
    new_idx2 = np.random.randint(0, n_providers, additional)
    new_mask = new_idx1 != new_idx2
    idx1 = np.concatenate([idx1, new_idx1[new_mask]])
    idx2 = np.concatenate([idx2, new_idx2[new_mask]])

idx1 = idx1[:n_pairs]
idx2 = idx2[:n_pairs]

print(f"Generated {len(idx1):,} pairs")

# Step 4: Calculate metrics for each pair
print("\nCalculating metrics for each pair...")

results = []

# Pre-extract tower1 columns once
tower1_cols = [col for col in embedding_df_filtered.columns if col.startswith('tower1_proc_emb_')]

for i in range(n_pairs):
    pin_a = filtered_pin_list[idx1[i]]
    pin_b = filtered_pin_list[idx2[i]]
    
    # Get provider names from pin_to_label
    name_a = pin_to_label.get(pin_a, 'Unknown')
    name_b = pin_to_label.get(pin_b, 'Unknown')
    
    # Get procedure sets
    procs_a = pin_to_procedures[pin_a]
    procs_b = pin_to_procedures[pin_b]
    
    # Calculate common procedures
    common_procs = procs_a & procs_b
    n_common = len(common_procs)
    n_total_a = len(procs_a)
    n_total_b = len(procs_b)
    
    # Calculate raw cosine similarity from proc_matrix
    vec_a = proc_matrix_filtered[idx1[i]:idx1[i]+1]
    vec_b = proc_matrix_filtered[idx2[i]:idx2[i]+1]
    raw_cosine = cosine_similarity(vec_a, vec_b)[0, 0]
    
    # Calculate embedding similarity (Tower 1)
    emb_a = embedding_df_filtered.iloc[idx1[i]][tower1_cols].values
    emb_b = embedding_df_filtered.iloc[idx2[i]][tower1_cols].values
    emb_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    
    results.append({
        'PIN_A': pin_a,
        'Provider_A_Name': name_a,
        'PIN_B': pin_b,
        'Provider_B_Name': name_b,
        'Common_Procedures': n_common,
        'Total_Procedures_A': n_total_a,
        'Total_Procedures_B': n_total_b,
        'Raw_Cosine_Similarity': raw_cosine,
        'Embedding_Similarity': emb_sim
    })
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1:,} pairs...")

results_df = pd.DataFrame(results)

print(f"\nResults calculated for {len(results_df):,} pairs")
print("\nSample results:")
print(results_df.head(10))

# Step 5: Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(results_df[['Common_Procedures', 'Total_Procedures_A', 'Total_Procedures_B', 
                   'Raw_Cosine_Similarity', 'Embedding_Similarity']].describe())

# Step 6: Create bins by common procedures and plot
print("\nCreating visualization...")

# Create bins for common procedures
bins = [0, 5, 10, 20, 50, 100, results_df['Common_Procedures'].max() + 1]
labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '100+']
results_df['Common_Proc_Bin'] = pd.cut(results_df['Common_Procedures'], bins=bins, labels=labels, right=False)

# Calculate average similarities per bin
bin_stats = results_df.groupby('Common_Proc_Bin', observed=True).agg({
    'Raw_Cosine_Similarity': 'mean',
    'Embedding_Similarity': 'mean',
    'Common_Procedures': 'count'
}).reset_index()
bin_stats.columns = ['Common_Proc_Bin', 'Avg_Raw_Cosine', 'Avg_Embedding_Sim', 'Pair_Count']

print("\nAverage similarities by common procedure bins:")
print(bin_stats)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter plot - Common Procedures vs Similarities
ax1 = axes[0, 0]
ax1.scatter(results_df['Common_Procedures'], results_df['Raw_Cosine_Similarity'], 
           alpha=0.3, s=10, label='Raw Cosine', color='blue')
ax1.scatter(results_df['Common_Procedures'], results_df['Embedding_Similarity'], 
           alpha=0.3, s=10, label='Embedding', color='red')
ax1.set_xlabel('Number of Common Procedures', fontsize=12)
ax1.set_ylabel('Similarity Score', fontsize=12)
ax1.set_title('Common Procedures vs Similarity Scores', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart - Average similarities by bin
ax2 = axes[0, 1]
x = np.arange(len(bin_stats))
width = 0.35
ax2.bar(x - width/2, bin_stats['Avg_Raw_Cosine'], width, label='Raw Cosine', color='blue', alpha=0.7)
ax2.bar(x + width/2, bin_stats['Avg_Embedding_Sim'], width, label='Embedding', color='red', alpha=0.7)
ax2.set_xlabel('Common Procedures Bin', fontsize=12)
ax2.set_ylabel('Average Similarity', fontsize=12)
ax2.set_title('Average Similarity by Common Procedure Bins', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(bin_stats['Common_Proc_Bin'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Raw Cosine vs Embedding Similarity
ax3 = axes[1, 0]
scatter = ax3.scatter(results_df['Raw_Cosine_Similarity'], results_df['Embedding_Similarity'], 
           alpha=0.3, s=10, c=results_df['Common_Procedures'], cmap='viridis')
ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
ax3.set_xlabel('Raw Cosine Similarity', fontsize=12)
ax3.set_ylabel('Embedding Similarity', fontsize=12)
ax3.set_title('Raw Cosine vs Embedding Similarity\n(colored by # common procedures)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Common Procedures')

# Plot 4: Distribution of common procedures
ax4 = axes[1, 1]
ax4.hist(results_df['Common_Procedures'], bins=50, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Number of Common Procedures', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Distribution of Common Procedures', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('step1_simple_validation_10k_pairs.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'step1_simple_validation_10k_pairs.png'")

plt.show()

# Step 7: Correlation analysis
from scipy.stats import pearsonr, spearmanr

print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

r1, p1 = pearsonr(results_df['Common_Procedures'], results_df['Raw_Cosine_Similarity'])
print(f"\nCommon Procedures vs Raw Cosine:")
print(f"  Pearson r = {r1:.4f} (p = {p1:.2e})")

r2, p2 = pearsonr(results_df['Common_Procedures'], results_df['Embedding_Similarity'])
print(f"\nCommon Procedures vs Embedding Similarity:")
print(f"  Pearson r = {r2:.4f} (p = {p2:.2e})")

r3, p3 = pearsonr(results_df['Raw_Cosine_Similarity'], results_df['Embedding_Similarity'])
print(f"\nRaw Cosine vs Embedding Similarity:")
print(f"  Pearson r = {r3:.4f} (p = {p3:.2e})")

print("\n" + "="*60)
print("Step 1 Complete!")
print("="*60)

# Save results
results_df.to_csv('step1_validation_results_10k.csv', index=False)
print("\nResults saved to 'step1_validation_results_10k.csv'")
