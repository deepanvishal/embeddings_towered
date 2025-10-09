import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_provider_codes(pin, code_type, top_n=10):
    """Extract top procedures or diagnoses for specified provider with memory optimization."""
    mask = (amt_smry_df['PIN'] == pin) & (amt_smry_df['code_type'] == code_type)
    codes = amt_smry_df.loc[mask, ['code', 'code_desc', 'claims']]
    
    if len(codes) == 0:
        if code_type == 'Procedure':
            return pd.DataFrame(columns=['Procedure_Code', 'Procedure_Description', 'Claims', '%'])
        else:
            return pd.DataFrame(columns=['ICD10_Code', 'ICD10_Description', 'Claims', '%'])
    
    grouped = codes.groupby(['code', 'code_desc'], as_index=False)['claims'].sum()
    total_claims = grouped['claims'].sum()
    grouped['pct'] = (grouped['claims'] / total_claims * 100).round(2)
    
    result = grouped.nlargest(top_n, 'claims')
    
    if code_type == 'Procedure':
        result = result.rename(columns={
            'code': 'Procedure_Code',
            'code_desc': 'Procedure_Description', 
            'claims': 'Claims',
            'pct': '%'
        })
    else:
        result = result.rename(columns={
            'code': 'ICD10_Code',
            'code_desc': 'ICD10_Description',
            'claims': 'Claims', 
            'pct': '%'
        })
    
    return result

def find_common_codes(pin_a, pin_b, code_type, top_n=10):
    """Identify common procedures/diagnoses between two providers."""
    codes_a = get_provider_codes(pin_a, code_type, 50)  # Get larger set for comparison
    codes_b = get_provider_codes(pin_b, code_type, 50)
    
    if len(codes_a) == 0 or len(codes_b) == 0:
        if code_type == 'Procedure':
            return pd.DataFrame(columns=['Procedure_Description', 'Claims_A', 'Claims_B', '%_A', '%_B'])
        else:
            return pd.DataFrame(columns=['ICD10_Description', 'Claims_A', 'Claims_B', '%_A', '%_B'])
    
    # Find intersection based on description
    desc_col = 'Procedure_Description' if code_type == 'Procedure' else 'ICD10_Description'
    common = codes_a.merge(codes_b, on=desc_col, suffixes=('_A', '_B'), how='inner')
    common['Total_Claims'] = common['Claims_A'] + common['Claims_B']
    
    result_cols = [desc_col, 'Claims_A', 'Claims_B', '%_A', '%_B']
    return common.nlargest(top_n, 'Total_Claims')[result_cols]

def get_provider_demographics(pin):
    """Extract demographic percentages for specified provider."""
    demo_cols = ['peds_pct', 'adults_pct', 'seniors_pct', 'Female_pct', 'Inpatient_pct', 'Emergency_pct']
    mask = member_df['PIN'] == pin
    
    if not mask.any():
        return pd.Series(index=demo_cols, dtype=float)
    
    return member_df.loc[mask, demo_cols].iloc[0]

def get_provider_costs(pin):
    """Extract cost distribution for specified provider."""
    cost_cols = [col for col in member_df.columns if col.startswith('med_cost_ctg_cd_')]
    mask = member_df['PIN'] == pin
    
    if not mask.any():
        return pd.Series(index=cost_cols, dtype=float)
    
    return member_df.loc[mask, cost_cols].iloc[0]

def create_comparison_table(data_a, data_b, name_a, name_b, metric_name):
    """Generate standardized comparison table between two providers."""
    if len(data_a) == 0 or len(data_b) == 0:
        return pd.DataFrame()
    
    comparison = pd.DataFrame({
        'metric': data_a.index,
        name_a: data_a.values,
        name_b: data_b.values
    })
    comparison['difference'] = (comparison[name_a] - comparison[name_b]).round(2)
    
    return comparison

def generate_pca_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Create PCA visualization that preserves centroid distances with memory cleanup."""
    plt.clf()
    plt.figure(figsize=(14, 8))
    
    # Get embeddings for the two groups
    same_mask = df['label'] == outlier_label
    other_mask = df['label'] == overlap_label
    
    # Combine embeddings for PCA
    group_embeddings = np.vstack([
        embeddings[same_mask.values],
        embeddings[other_mask.values]
    ])
    
    # Fit PCA on combined group data
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(group_embeddings)
    
    # Split back into groups
    same_count = same_mask.sum()
    pca_same = pca_coords[:same_count]
    pca_other = pca_coords[same_count:]
    
    # Calculate actual centroids in embedding space, then transform to PCA space
    same_centroid_emb = embeddings[same_mask.values].mean(axis=0)
    other_centroid_emb = embeddings[other_mask.values].mean(axis=0)
    
    centroids_pca = pca.transform([same_centroid_emb, other_centroid_emb])
    same_centroid_pca = centroids_pca[0]
    other_centroid_pca = centroids_pca[1]
    
    # Plot group points
    plt.scatter(pca_same[:, 0], pca_same[:, 1], alpha=0.6, label=outlier_label, s=30, color='lightblue')
    plt.scatter(pca_other[:, 0], pca_other[:, 1], alpha=0.6, label=overlap_label, s=30, color='lightcoral')
    
    # Plot actual centroids (preserves true distances)
    plt.scatter(same_centroid_pca[0], same_centroid_pca[1], marker='X', s=400, 
                color='blue', edgecolors='black', linewidth=2, label=f'{outlier_label} Centroid')
    plt.scatter(other_centroid_pca[0], other_centroid_pca[1], marker='X', s=400, 
                color='red', edgecolors='black', linewidth=2, label=f'{overlap_label} Centroid')
    
    # Transform and plot key providers
    key_providers = [
        (outlier_pin, '*', 'darkred', 'Outlier Provider', 300),
        (typical_same_pin, 'o', 'darkblue', f'Typical {outlier_label}', 200),
        (typical_other_pin, 's', 'darkgreen', f'Typical {overlap_label}', 200)
    ]
    
    for pin, marker, color, label_text, size in key_providers:
        provider_idx = df[df['PIN'] == pin].index
        if len(provider_idx) > 0:
            provider_emb = embeddings[provider_idx[0] - df.index[0]]  # Adjust for index
            provider_pca = pca.transform([provider_emb])[0]
            
            plt.scatter(provider_pca[0], provider_pca[1], marker=marker, s=size, 
                       color=color, edgecolors='black', linewidth=1, label=label_text)
            
            # Draw lines to both centroids for outlier
            if pin == outlier_pin:
                plt.plot([provider_pca[0], same_centroid_pca[0]], 
                        [provider_pca[1], same_centroid_pca[1]], 
                        'b--', alpha=0.7, linewidth=2, label='Distance to Own Centroid')
                plt.plot([provider_pca[0], other_centroid_pca[0]], 
                        [provider_pca[1], other_centroid_pca[1]], 
                        'r--', alpha=0.7, linewidth=2, label='Distance to Other Centroid')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'PCA Visualization: {outlier_label} vs {overlap_label}\n(Preserves True Centroid Distances)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Memory cleanup
    plt.close('all')
    del group_embeddings, pca_coords, pca_same, pca_other
    gc.collect()

def generate_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Create both t-SNE and PCA visualizations."""
    print("t-SNE Visualization (Neighborhood Preservation):")
    generate_tsne_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label)
    
    print("\nPCA Visualization (Distance Preservation):")
    generate_pca_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label)

def generate_tsne_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Create t-SNE visualization with centroids and memory cleanup."""
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    # Filter data efficiently
    same_mask = tsne_df['label'] == outlier_label
    other_mask = tsne_df['label'] == overlap_label
    
    same_data = tsne_df.loc[same_mask]
    other_data = tsne_df.loc[other_mask]
    
    plt.scatter(same_data['tsne_x'], same_data['tsne_y'], 
                alpha=0.6, label=outlier_label, s=30)
    plt.scatter(other_data['tsne_x'], other_data['tsne_y'], 
                alpha=0.6, label=overlap_label, s=30)
    
    # Calculate and plot t-SNE centroids (approximate)
    same_centroid_x = same_data['tsne_x'].mean()
    same_centroid_y = same_data['tsne_y'].mean()
    other_centroid_x = other_data['tsne_x'].mean()
    other_centroid_y = other_data['tsne_y'].mean()
    
    plt.scatter(same_centroid_x, same_centroid_y, marker='X', s=300, 
                color='blue', edgecolors='black', linewidth=2, label=f'{outlier_label} t-SNE Center')
    plt.scatter(other_centroid_x, other_centroid_y, marker='X', s=300, 
                color='orange', edgecolors='black', linewidth=2, label=f'{overlap_label} t-SNE Center')
    
    # Highlight key providers
    for pin, marker, color, label_suffix in [
        (outlier_pin, '*', 'red', 'Outlier'),
        (typical_same_pin, 'o', 'blue', f'Typical {outlier_label}'),
        (typical_other_pin, 's', 'green', f'Typical {overlap_label}')
    ]:
        provider_data = tsne_df[tsne_df['PIN'] == pin]
        if len(provider_data) > 0:
            plt.scatter(provider_data['tsne_x'], provider_data['tsne_y'], 
                       marker=marker, s=200, color=color, edgecolors='black', 
                       linewidth=1, label=label_suffix)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE: {outlier_label} vs {overlap_label} (Neighborhood View)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.show()
    
    # Memory cleanup
    plt.close('all')
    gc.collect()

# Create interface elements
dropdown_options = [("Select overlapping provider analysis", None)]
for idx, row in analysis_df.iterrows():
    option_text = f"{row['outlier_label']} vs {row['overlap_label']} - {row['outlier_name']} vs {row['typical_other_name']}"
    dropdown_options.append((option_text, idx))

analysis_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Overlapping Provider Analysis:',
    style={'description_width': 'initial'},
    layout={'width': '800px'}
)

output_area = widgets.Output()

def execute_analysis(change):
    """Primary analysis execution handler with memory management."""
    if change['new'] is None:
        return
    
    with output_area:
        clear_output(wait=True)
        
        # Memory efficient data access
        selected_idx = change['new']
        selected = analysis_df.iloc[selected_idx]
        
        # Analysis header
        print("=" * 80)
        print(f"OVERLAPPING PROVIDER ANALYSIS: {selected['outlier_name']}")
        print("=" * 80)
        
        print(f"Selected Provider for Analysis: {selected['outlier_name']} ({selected['outlier_label']})")
        print(f"The provider embeddings indicate the provider is close to {selected['overlap_label']}.")
        print(f"The typical provider (closest to centroid) of the {selected['overlap_label']} is {selected['typical_other_name']}")
        print(f"The typical provider (closest to centroid) of the {selected['outlier_label']} is {selected['typical_same_name']}")
        
        print(f"\nDistance Metrics:")
        print(f"  Distance to own centroid ({selected['outlier_label']}): {selected['own_distance']:.4f}")
        print(f"  Distance to overlapping centroid ({selected['overlap_label']}): {selected['other_distance']:.4f}")
        print(f"  Overlap coefficient: {selected['overlap_score']:.4f}")
        
        # Within-group analysis
        print(f"\n{'='*70}")
        print(f"WITHIN-GROUP ANALYSIS: {selected['outlier_label']} Providers")
        print(f"{'='*70}")
        
        # Procedure analysis
        proc_outlier = get_provider_codes(selected['outlier_pin'], 'Procedure', 10)
        proc_typical_same = get_provider_codes(selected['typical_same_pin'], 'Procedure', 10)
        common_proc_1 = find_common_codes(selected['outlier_pin'], selected['typical_same_pin'], 'Procedure', 10)
        
        print(f"\nProcedure Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top 10 Procedures:")
        if len(proc_outlier) > 0:
            display(proc_outlier)
        else:
            print("No procedure data available")
        
        print(f"\n{selected['typical_same_name']} - Top 10 Procedures:")
        if len(proc_typical_same) > 0:
            display(proc_typical_same)
        else:
            print("No procedure data available")
        
        print(f"\nTop 10 Common Procedures:")
        if len(common_proc_1) > 0:
            display(common_proc_1)
        else:
            print("No common procedures found in top sets")
        
        # Diagnosis analysis
        diag_outlier = get_provider_codes(selected['outlier_pin'], 'Diagnosis', 10)
        diag_typical_same = get_provider_codes(selected['typical_same_pin'], 'Diagnosis', 10)
        common_diag_1 = find_common_codes(selected['outlier_pin'], selected['typical_same_pin'], 'Diagnosis', 10)
        
        print(f"\nDiagnosis Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top 10 Diagnoses:")
        if len(diag_outlier) > 0:
            display(diag_outlier)
        else:
            print("No diagnosis data available")
        
        print(f"\n{selected['typical_same_name']} - Top 10 Diagnoses:")
        if len(diag_typical_same) > 0:
            display(diag_typical_same)
        else:
            print("No diagnosis data available")
        
        print(f"\nTop 10 Common Diagnoses:")
        if len(common_diag_1) > 0:
            display(common_diag_1)
        else:
            print("No common diagnoses found in top sets")
        
        # Demographics analysis
        demo_outlier = get_provider_demographics(selected['outlier_pin'])
        demo_typical_same = get_provider_demographics(selected['typical_same_pin'])
        demo_comparison_1 = create_comparison_table(
            demo_outlier, demo_typical_same, 
            selected['outlier_name'], selected['typical_same_name'], 'demographics'
        )
        
        print(f"\nDemographic Distribution Analysis:")
        if len(demo_comparison_1) > 0:
            display(demo_comparison_1)
        
        # Cost analysis
        cost_outlier = get_provider_costs(selected['outlier_pin'])
        cost_typical_same = get_provider_costs(selected['typical_same_pin'])
        cost_comparison_1 = create_comparison_table(
            cost_outlier, cost_typical_same,
            selected['outlier_name'], selected['typical_same_name'], 'costs'
        )
        
        print(f"\nCost Distribution Analysis:")
        if len(cost_comparison_1) > 0:
            display(cost_comparison_1)
        
        # Cross-group analysis
        print(f"\n{'='*70}")
        print(f"CROSS-GROUP ANALYSIS: {selected['outlier_label']} vs {selected['overlap_label']}")
        print(f"{'='*70}")
        
        # Procedure analysis
        proc_typical_other = get_provider_codes(selected['typical_other_pin'], 'Procedure', 10)
        common_proc_2 = find_common_codes(selected['outlier_pin'], selected['typical_other_pin'], 'Procedure', 10)
        
        print(f"\nProcedure Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top 10 Procedures:")
        if len(proc_outlier) > 0:
            display(proc_outlier)
        
        print(f"\n{selected['typical_other_name']} - Top 10 Procedures:")
        if len(proc_typical_other) > 0:
            display(proc_typical_other)
        else:
            print("No procedure data available")
        
        print(f"\nTop 10 Common Procedures:")
        if len(common_proc_2) > 0:
            display(common_proc_2)
        else:
            print("No common procedures found in top sets")
        
        # Diagnosis analysis
        diag_typical_other = get_provider_codes(selected['typical_other_pin'], 'Diagnosis', 10)
        common_diag_2 = find_common_codes(selected['outlier_pin'], selected['typical_other_pin'], 'Diagnosis', 10)
        
        print(f"\nDiagnosis Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top 10 Diagnoses:")
        if len(diag_outlier) > 0:
            display(diag_outlier)
        
        print(f"\n{selected['typical_other_name']} - Top 10 Diagnoses:")
        if len(diag_typical_other) > 0:
            display(diag_typical_other)
        else:
            print("No diagnosis data available")
        
        print(f"\nTop 10 Common Diagnoses:")
        if len(common_diag_2) > 0:
            display(common_diag_2)
        else:
            print("No common diagnoses found in top sets")
        
        # Demographics analysis
        demo_typical_other = get_provider_demographics(selected['typical_other_pin'])
        demo_comparison_2 = create_comparison_table(
            demo_outlier, demo_typical_other,
            selected['outlier_name'], selected['typical_other_name'], 'demographics'
        )
        
        print(f"\nDemographic Distribution Analysis:")
        if len(demo_comparison_2) > 0:
            display(demo_comparison_2)
        
        # Cost analysis
        cost_typical_other = get_provider_costs(selected['typical_other_pin'])
        cost_comparison_2 = create_comparison_table(
            cost_outlier, cost_typical_other,
            selected['outlier_name'], selected['typical_other_name'], 'costs'
        )
        
        print(f"\nCost Distribution Analysis:")
        if len(cost_comparison_2) > 0:
            display(cost_comparison_2)
        
        # Visualization
        print(f"\nProvider Positioning Visualization:")
        generate_visualization(
            selected['outlier_pin'], selected['typical_same_pin'], selected['typical_other_pin'],
            selected['outlier_label'], selected['overlap_label']
        )
        
        # Memory cleanup
        del proc_outlier, proc_typical_same, proc_typical_other
        del diag_outlier, diag_typical_same, diag_typical_other
        del common_proc_1, common_proc_2, common_diag_1, common_diag_2
        del demo_outlier, demo_typical_same, demo_typical_other
        del cost_outlier, cost_typical_same, cost_typical_other
        gc.collect()

analysis_dropdown.observe(execute_analysis, names='value')

# Interface deployment
print("OVERLAPPING PROVIDER ANALYSIS SYSTEM")
print("=" * 50)
print(f"Available overlapping provider cases: {len(analysis_df)}")
print("Memory-optimized implementation with professional reporting")

display(widgets.VBox([analysis_dropdown, output_area]))

print("\nSystem ready for overlapping provider analysis execution.")