import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import gc

def create_pairwise_embedding_plots():
    """Create simple pairwise plots showing embedding differences across 64 dimensions."""
    
    print("Creating pairwise embedding difference plots...")
    
    # Calculate centroids for each label
    all_labels = sorted(df['label'].unique())
    label_centroids = {}
    
    for label in all_labels:
        mask = df['label'] == label
        label_centroids[label] = embeddings[mask.values].mean(axis=0)
    
    # Calculate pairwise differences
    difference_data = {}
    separation_scores = {}
    
    for label_a, label_b in combinations(all_labels, 2):
        centroid_a = label_centroids[label_a]
        centroid_b = label_centroids[label_b]
        
        # Calculate absolute difference across all 64 dimensions
        diff_vector = np.abs(centroid_a - centroid_b)
        pair_name = f"{label_a} vs {label_b}"
        
        difference_data[pair_name] = diff_vector
        separation_scores[pair_name] = diff_vector.mean()
    
    # Sort pairs by separation score
    sorted_pairs = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create plots
    n_pairs = len(sorted_pairs)
    cols = 3
    rows = (n_pairs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    x_positions = range(64)
    
    for i, (pair_name, separation_score) in enumerate(sorted_pairs):
        if i >= len(axes):
            break
            
        diff_vector = difference_data[pair_name]
        
        # Create bar plot
        axes[i].bar(x_positions, diff_vector, alpha=0.7, color='steelblue')
        axes[i].set_xlabel('Embedding Dimension (0-63)')
        axes[i].set_ylabel('Absolute Difference')
        axes[i].set_title(f'{pair_name}\nSeparation Score: {separation_score:.4f}')
        axes[i].grid(True, alpha=0.3)
        
        # Highlight top 5 dimensions with highest differences
        top_dims = np.argsort(diff_vector)[-5:]
        for dim in top_dims:
            axes[i].bar(dim, diff_vector[dim], color='red', alpha=0.8)
    
    # Hide unused subplots
    for i in range(len(sorted_pairs), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    gc.collect()
    
    # Return data for further analysis
    return difference_data, separation_scores

# Execute
difference_data, separation_scores = create_pairwise_embedding_plots()

print(f"\n📊 PAIRWISE EMBEDDING ANALYSIS:")
print(f"Total pairs: {len(difference_data)}")
print(f"\nTop 5 best separated pairs:")
sorted_pairs = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
for pair, score in sorted_pairs[:5]:
    print(f"  {pair}: {score:.4f}")
print(f"\nBottom 5 worst separated pairs:")
for pair, score in sorted_pairs[-5:]:
    print(f"  {pair}: {score:.4f}")
