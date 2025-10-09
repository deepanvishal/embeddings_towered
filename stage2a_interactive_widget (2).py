import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import gc

def get_provider_procedures(pin, top_n=10):
    """Extract top procedures for specified provider with memory optimization."""
    mask = (amt_smry_df['PIN'] == pin) & (amt_smry_df['code_type'] == 'Procedure')
    procedures = amt_smry_df.loc[mask, ['code', 'code_desc', 'claims']]
    
    if len(procedures) == 0:
        return pd.DataFrame(columns=['code_desc', 'claims', 'pct'])
    
    grouped = procedures.groupby(['code', 'code_desc'], as_index=False)['claims'].sum()
    total_claims = grouped['claims'].sum()
    grouped['pct'] = (grouped['claims'] / total_claims * 100).round(2)
    
    return grouped.nlargest(top_n, 'claims')[['code_desc', 'claims', 'pct']]

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

def generate_visualization(outlier_pin, typical_same_pin, typical_other_pin, outlier_label, overlap_label):
    """Create t-SNE visualization with memory cleanup."""
    plt.clf()
    plt.figure(figsize=(10, 6))
    
    # Filter data efficiently
    same_mask = tsne_df['label'] == outlier_label
    other_mask = tsne_df['label'] == overlap_label
    
    plt.scatter(tsne_df.loc[same_mask, 'tsne_x'], tsne_df.loc[same_mask, 'tsne_y'], 
                alpha=0.6, label=outlier_label, s=30)
    plt.scatter(tsne_df.loc[other_mask, 'tsne_x'], tsne_df.loc[other_mask, 'tsne_y'], 
                alpha=0.6, label=overlap_label, s=30)
    
    # Highlight key providers
    for pin, marker, color, label_suffix in [
        (outlier_pin, '*', 'red', 'Outlier'),
        (typical_same_pin, 'o', 'blue', f'Typical {outlier_label}'),
        (typical_other_pin, 's', 'green', f'Typical {overlap_label}')
    ]:
        provider_data = tsne_df[tsne_df['PIN'] == pin]
        if len(provider_data) > 0:
            plt.scatter(provider_data['tsne_x'], provider_data['tsne_y'], 
                       marker=marker, s=200, color=color, label=label_suffix)
    
    plt.legend()
    plt.title(f'{outlier_label} vs {overlap_label} Provider Positioning')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.show()
    
    # Memory cleanup
    plt.close('all')
    gc.collect()

# Create interface elements
dropdown_options = [("Select analysis case", None)]
for idx, row in analysis_df.iterrows():
    dropdown_options.append((row['analysis_description'], idx))

analysis_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Analysis Selection:',
    style={'description_width': 'initial'},
    layout={'width': '700px'}
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
        print(f"PROVIDER OVERLAP ANALYSIS: {selected['outlier_name']}")
        print("=" * 80)
        
        print(f"Outlier Provider: {selected['outlier_name']} ({selected['outlier_label']})")
        print(f"Benchmark Same Group: {selected['typical_same_name']} ({selected['outlier_label']})")
        print(f"Benchmark Other Group: {selected['typical_other_name']} ({selected['overlap_label']})")
        
        print(f"\nDistance Metrics:")
        print(f"  Distance to own centroid: {selected['own_distance']:.4f}")
        print(f"  Distance to other centroid: {selected['other_distance']:.4f}")
        print(f"  Overlap coefficient: {selected['overlap_score']:.4f}")
        
        # Within-group analysis
        print(f"\n{'='*70}")
        print(f"WITHIN-GROUP ANALYSIS: {selected['outlier_label']} Providers")
        print(f"{'='*70}")
        
        # Procedure analysis
        proc_outlier = get_provider_procedures(selected['outlier_pin'])
        proc_typical_same = get_provider_procedures(selected['typical_same_pin'])
        
        print(f"\nProcedure Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top Procedures:")
        if len(proc_outlier) > 0:
            display(proc_outlier.head())
        else:
            print("No procedure data available")
        
        print(f"\n{selected['typical_same_name']} - Top Procedures:")
        if len(proc_typical_same) > 0:
            display(proc_typical_same.head())
        else:
            print("No procedure data available")
        
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
        proc_typical_other = get_provider_procedures(selected['typical_other_pin'])
        
        print(f"\nProcedure Volume Analysis:")
        print(f"\n{selected['outlier_name']} - Top Procedures:")
        if len(proc_outlier) > 0:
            display(proc_outlier.head())
        
        print(f"\n{selected['typical_other_name']} - Top Procedures:")
        if len(proc_typical_other) > 0:
            display(proc_typical_other.head())
        
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
        del demo_outlier, demo_typical_same, demo_typical_other
        del cost_outlier, cost_typical_same, cost_typical_other
        gc.collect()

analysis_dropdown.observe(execute_analysis, names='value')

# Interface deployment
print("PROVIDER OVERLAP ANALYSIS SYSTEM")
print("=" * 45)
print(f"Available analysis cases: {len(analysis_df)}")
print("Memory-optimized implementation with professional reporting")

display(widgets.VBox([analysis_dropdown, output_area]))

print("\nSystem ready for analysis execution.")