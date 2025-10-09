import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def create_embedding_comparison_chart(outlier_pin, typical_same_pin, typical_other_pin, outlier_name, typical_same_name, typical_other_name):
    """Create embedding comparison chart."""
    
    # Get embeddings for the three providers
    outlier_idx = df[df['PIN'] == outlier_pin].index[0]
    typical_same_idx = df[df['PIN'] == typical_same_pin].index[0] 
    typical_other_idx = df[df['PIN'] == typical_other_pin].index[0]
    
    outlier_emb = embeddings[outlier_idx - df.index[0]]
    typical_same_emb = embeddings[typical_same_idx - df.index[0]]
    typical_other_emb = embeddings[typical_other_idx - df.index[0]]
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    x_positions = range(64)
    plt.plot(x_positions, outlier_emb, marker='o', linewidth=2, markersize=4, 
             label=f'{outlier_name} (Overlapping Provider)', color='red', alpha=0.8)
    plt.plot(x_positions, typical_same_emb, marker='s', linewidth=2, markersize=4,
             label=f'{typical_same_name} (Same Group)', color='blue', alpha=0.8)
    plt.plot(x_positions, typical_other_emb, marker='^', linewidth=2, markersize=4,
             label=f'{typical_other_name} (Other Group)', color='green', alpha=0.8)
    
    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Embedding Value', fontsize=12)
    plt.title('Provider Embedding Comparison: 64-Dimensional Profile', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to temp file
    temp_filename = f"temp_embedding_{outlier_pin}.png"
    plt.savefig(temp_filename, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return temp_filename

def create_pca_chart(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label, outlier_name, typical_same_name, typical_other_name):
    """Create PCA visualization chart."""
    
    # Get embeddings for the two groups
    same_mask = df['label'] == outlier_label
    other_mask = df['label'] == overlap_label
    
    # Combine embeddings for PCA
    group_embeddings = np.vstack([
        embeddings[same_mask.values],
        embeddings[other_mask.values]
    ])
    
    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(group_embeddings)
    
    # Split back into groups
    same_count = same_mask.sum()
    pca_same = pca_coords[:same_count]
    pca_other = pca_coords[same_count:]
    
    # Calculate centroids
    same_centroid_emb = embeddings[same_mask.values].mean(axis=0)
    other_centroid_emb = embeddings[other_mask.values].mean(axis=0)
    centroids_pca = pca.transform([same_centroid_emb, other_centroid_emb])
    same_centroid_pca = centroids_pca[0]
    other_centroid_pca = centroids_pca[1]
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot groups
    plt.scatter(pca_same[:, 0], pca_same[:, 1], alpha=0.6, label=outlier_label.title(), s=30, color='lightblue')
    plt.scatter(pca_other[:, 0], pca_other[:, 1], alpha=0.6, label=overlap_label.title(), s=30, color='lightcoral')
    
    # Plot centroids
    plt.scatter(same_centroid_pca[0], same_centroid_pca[1], marker='X', s=400, 
                color='blue', edgecolors='black', linewidth=2, label=f'{outlier_label.title()} Centroid')
    plt.scatter(other_centroid_pca[0], other_centroid_pca[1], marker='X', s=400, 
                color='red', edgecolors='black', linewidth=2, label=f'{overlap_label.title()} Centroid')
    
    # Plot key providers
    key_providers = [
        (outlier_pin, '*', 'darkred', f'{outlier_name}', 300),
        (typical_same_pin, 'o', 'darkblue', f'{outlier_label.title()} - {typical_same_name}', 200),
        (typical_other_pin, 's', 'darkgreen', f'{overlap_label.title()} - {typical_other_name}', 200)
    ]
    
    for pin, marker, color, label_text, size in key_providers:
        provider_idx = df[df['PIN'] == pin].index
        if len(provider_idx) > 0:
            provider_emb = embeddings[provider_idx[0] - df.index[0]]
            provider_pca = pca.transform([provider_emb])[0]
            
            plt.scatter(provider_pca[0], provider_pca[1], marker=marker, s=size, 
                       color=color, edgecolors='black', linewidth=1, label=label_text)
            
            if pin == outlier_pin:
                plt.plot([provider_pca[0], same_centroid_pca[0]], 
                        [provider_pca[1], same_centroid_pca[1]], 
                        'b--', alpha=0.7, linewidth=2, label='Distance to Own Centroid')
                plt.plot([provider_pca[0], other_centroid_pca[0]], 
                        [provider_pca[1], other_centroid_pca[1]], 
                        'r--', alpha=0.7, linewidth=2, label='Distance to Other Centroid')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'PCA Visualization: {outlier_label.title()} vs {overlap_label.title()}\n(Preserves True Centroid Distances)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to temp file
    temp_filename = f"temp_pca_{outlier_pin}.png"
    plt.savefig(temp_filename, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Memory cleanup
    del group_embeddings, pca_coords, pca_same, pca_other
    gc.collect()
    
    return temp_filename

print("✅ Step 3 Complete: Chart functions ready")