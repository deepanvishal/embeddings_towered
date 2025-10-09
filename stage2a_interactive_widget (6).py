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
            return pd.DataFrame(columns=['Procedure_Code', 'Procedure_Description', 'Claims_A', 'Claims_B', '%_A', '%_B'])
        else:
            return pd.DataFrame(columns=['ICD10_Code', 'ICD10_Description', 'Claims_A', 'Claims_B', '%_A', '%_B'])
    
    # Find intersection based on description and code
    desc_col = 'Procedure_Description' if code_type == 'Procedure' else 'ICD10_Description'
    code_col = 'Procedure_Code' if code_type == 'Procedure' else 'ICD10_Code'
    
    common = codes_a.merge(codes_b, on=[code_col, desc_col], suffixes=('_A', '_B'), how='inner')
    common['Total_Claims'] = common['Claims_A'] + common['Claims_B']
    
    result_cols = [code_col, desc_col, 'Claims_A', 'Claims_B', '%_A', '%_B']
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
    cost_category_mapping = {
        'med_cost_ctg_cd_001_pct': 'IP Facility',
        'med_cost_ctg_cd_002_pct': 'AMB Facility',
        'med_cost_ctg_cd_003_pct': 'Emergency',
        'med_cost_ctg_cd_004_pct': 'Specialty Physician',
        'med_cost_ctg_cd_005_pct': 'PCP Physician',
        'med_cost_ctg_cd_006_pct': 'Radiology',
        'med_cost_ctg_cd_007_pct': 'LAB',
        'med_cost_ctg_cd_008_pct': 'Home Health',
        'med_cost_ctg_cd_009_pct': 'Mental Health',
        'med_cost_ctg_cd_010_pct': 'Medical Rx',
        'med_cost_ctg_cd_016_pct': 'Other'
    }
    
    cost_cols = list(cost_category_mapping.keys())
    mask = member_df['PIN'] == pin
    
    if not mask.any():
        return pd.Series(index=cost_cols, dtype=float)
    
    costs = member_df.loc[mask, cost_cols].iloc[0]
    # Map to readable names
    costs.index = [cost_category_mapping.get(col, col) for col in costs.index]
    
    return costs

def create_comparison_table(data_a, data_b, name_a, name_b, metric_name):
    """Generate standardized comparison table between two providers."""
    if len(data_a) == 0 or len(data_b) == 0:
        return pd.DataFrame()
    
    comparison = pd.DataFrame({
        'Metric': data_a.index,
        name_a: data_a.values,
        name_b: data_b.values
    })
    comparison['Difference'] = (comparison[name_a] - comparison[name_b]).round(2)
    
    return comparison

def generate_pca_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Create PCA visualization that preserves centroid distances with memory cleanup."""
    plt.clf()
    plt.figure(figsize=(14, 8))
    
    # Get provider names for legend
    outlier_name = new_scores_with_names[new_scores_with_names['PIN'] == outlier_pin]['PIN_name'].iloc[0]
    typical_same_name = new_scores_with_names[new_scores_with_names['PIN'] == typical_same_pin]['PIN_name'].iloc[0]
    typical_other_name = new_scores_with_names[new_scores_with_names['PIN'] == typical_other_pin]['PIN_name'].iloc[0]
    
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
    plt.scatter(pca_same[:, 0], pca_same[:, 1], alpha=0.6, label=outlier_label.title(), s=30, color='lightblue')
    plt.scatter(pca_other[:, 0], pca_other[:, 1], alpha=0.6, label=overlap_label.title(), s=30, color='lightcoral')
    
    # Plot actual centroids (preserves true distances)
    plt.scatter(same_centroid_pca[0], same_centroid_pca[1], marker='X', s=400, 
                color='blue', edgecolors='black', linewidth=2, label=f'{outlier_label.title()} Centroid')
    plt.scatter(other_centroid_pca[0], other_centroid_pca[1], marker='X', s=400, 
                color='red', edgecolors='black', linewidth=2, label=f'{overlap_label.title()} Centroid')
    
    # Transform and plot key providers
    key_providers = [
        (outlier_pin, '*', 'darkred', f'{outlier_name}', 300),
        (typical_same_pin, 'o', 'darkblue', f'{outlier_label.title()} - {typical_same_name}', 200),
        (typical_other_pin, 's', 'darkgreen', f'{overlap_label.title()} - {typical_other_name}', 200)
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
    plt.title(f'PCA Visualization: {outlier_label.title()} vs {overlap_label.title()}\n(Preserves True Centroid Distances)')
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
    """Create PCA visualization only."""
    print("PCA Visualization (Distance Preservation):")
    generate_pca_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label)

def generate_tsne_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Removed - t-SNE visualization no longer used."""
    pass

# Create interface elements
dropdown_options = [("Select overlapping provider analysis", None)]
for idx, row in analysis_df.iterrows():
    option_text = f"{row['outlier_label'].title()} vs {row['overlap_label'].title()} - {row['outlier_name']} vs {row['typical_other_name']}"
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
        
        print(f"Selected Provider for Analysis: {selected['outlier_name']} ({selected['outlier_label'].title()})")
        print(f"The provider embeddings indicate the provider is close to {selected['overlap_label'].title()}.")
        print(f"The provider closest to centroid of {selected['overlap_label'].title()} is {selected['typical_other_name']}")
        print(f"The provider closest to centroid of {selected['outlier_label'].title()} is {selected['typical_same_name']}")
        
        print(f"\nDistance Metrics:")
        print(f"  Distance to own centroid ({selected['outlier_label'].title()}): {selected['own_distance']:.4f}")
        print(f"  Distance to overlapping centroid ({selected['overlap_label'].title()}): {selected['other_distance']:.4f}")
        print(f"  Overlap coefficient: {selected['overlap_score']:.4f}")
        
        # Within-group analysis
        print(f"\n{'='*70}")
        print(f"WITHIN-GROUP ANALYSIS: {selected['outlier_label'].title()} Providers")
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
        print(f"CROSS-GROUP ANALYSIS: {selected['outlier_label'].title()} vs {selected['overlap_label'].title()}")
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