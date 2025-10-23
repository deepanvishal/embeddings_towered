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
