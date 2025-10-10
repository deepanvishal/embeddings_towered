import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

def create_embedding_pairplot():
    """Create 15x15 pairplot with 64-dimension bar charts in each tile."""
    
    print("Creating 15x15 embedding difference pairplot...")
    
    # Calculate centroids for each label
    all_labels = sorted(df['label'].unique())
    n_labels = len(all_labels)
    label_centroids = {}
    
    print(f"Processing {n_labels} labels...")
    for label in all_labels:
        mask = df['label'] == label
        label_centroids[label] = embeddings[mask.values].mean(axis=0)
    
    # Create 15x15 subplot grid
    fig, axes = plt.subplots(n_labels, n_labels, figsize=(20, 20))
    
    x_positions = range(64)
    
    for i in range(n_labels):
        for j in range(n_labels):
            ax = axes[i, j]
            
            if i > j:  # Lower triangle only
                label_a = all_labels[i]
                label_b = all_labels[j]
                
                # Calculate absolute difference
                centroid_a = label_centroids[label_a]
                centroid_b = label_centroids[label_b]
                diff_vector = np.abs(centroid_a - centroid_b)
                
                # Create mini bar plot
                ax.bar(x_positions, diff_vector, alpha=0.7, color='steelblue', width=1.0)
                
                # Highlight top 3 dimensions
                top_dims = np.argsort(diff_vector)[-3:]
                for dim in top_dims:
                    ax.bar(dim, diff_vector[dim], color='red', alpha=0.8, width=1.0)
                
                # Clean formatting
                ax.set_xlim(-1, 64)
                ax.set_ylim(0, diff_vector.max() * 1.1)
                
                # Remove tick labels for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add separation score as text
                separation_score = diff_vector.mean()
                ax.text(32, diff_vector.max() * 0.8, f'{separation_score:.3f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold')
                
            elif i == j:  # Diagonal - label names
                ax.text(0.5, 0.5, all_labels[i], ha='center', va='center', 
                       fontsize=10, fontweight='bold', transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                
            else:  # Upper triangle - hide
                ax.set_visible(False)
            
            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)
    
    # Add overall labels
    fig.suptitle('Label Centroid Separation Pairplot\n(64-Dimensional Embedding Differences)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='steelblue', alpha=0.7, label='Embedding Differences'),
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, label='Top 3 Dimensions'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.92))
    
    # Add axis labels
    fig.text(0.5, 0.02, 'Each tile shows absolute differences across 64 embedding dimensions', 
             ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Red bars highlight the 3 most discriminative dimensions', 
             va='center', rotation=90, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    plt.show()
    
    # Memory cleanup
    plt.close()
    gc.collect()
    
    print(f"✅ Pairplot complete!")
    print(f"- {n_labels}x{n_labels} grid with {(n_labels*(n_labels-1))//2} comparison tiles")
    print(f"- Each tile shows 64-dimensional difference profile")
    print(f"- Numbers in tiles = mean separation score")
    print(f"- Red bars = top 3 most discriminative dimensions")

# Execute
create_embedding_pairplot()