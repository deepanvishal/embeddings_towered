import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd

print("Stage 2A: Interactive Widget Creation")
print("=" * 40)

# Step 1: Prepare dropdown options from analysis_df
print("Step 1: Preparing dropdown options...")

dropdown_options = []
for _, row in analysis_df.iterrows():
    option_text = row['analysis_description']
    dropdown_options.append((option_text, row))

print(f"Created {len(dropdown_options)} analysis options")

# Step 2: Create the main selection widget
print("\nStep 2: Creating selection interface...")

# Main dropdown
analysis_dropdown = widgets.Dropdown(
    options=dropdown_options,
    value=None,
    description='Select Analysis:',
    style={'description_width': 'initial'},
    layout={'width': '700px'}
)

# Info display widget
info_output = widgets.Output()

# Analysis output widget  
analysis_output = widgets.Output()

# Step 3: Create selection handler
def on_analysis_selected(change):
    if change['new'] is None:
        return
        
    selected_row = change['new']
    
    # Clear previous outputs
    info_output.clear_output()
    analysis_output.clear_output()
    
    # Display selection info
    with info_output:
        print("=" * 60)
        print("SELECTED ANALYSIS")
        print("=" * 60)
        print(f"Outlier Provider: {selected_row['outlier_name']} ({selected_row['outlier_label']})")
        print(f"Typical Same Group: {selected_row['typical_same_name']} ({selected_row['outlier_label']})")
        print(f"Typical Other Group: {selected_row['typical_other_name']} ({selected_row['overlap_label']})")
        print(f"\nDistance Context:")
        print(f"  Distance to own centroid ({selected_row['outlier_label']}): {selected_row['own_distance']:.4f}")
        print(f"  Distance to other centroid ({selected_row['overlap_label']}): {selected_row['other_distance']:.4f}")
        print(f"  Overlap Score: {selected_row['overlap_score']:.4f}")
        print(f"\nComparison: {selected_row['comparison_summary']}")
        print("=" * 60)
    
    # Trigger detailed analysis
    with analysis_output:
        print("Loading detailed analysis...")
        run_detailed_analysis(selected_row)

# Step 4: Import the detailed analysis function from Stage 2B
def run_detailed_analysis(selected_row):
    """Full detailed analysis with actual data comparisons"""
    return run_detailed_analysis_enhanced(selected_row)

# Step 5: Connect the handler
analysis_dropdown.observe(on_analysis_selected, names='value')

# Step 6: Create summary statistics display
def create_summary_widget():
    summary_output = widgets.Output()
    
    with summary_output:
        print("OVERLAP ANALYSIS SUMMARY")
        print("=" * 40)
        
        # Display overlap summary
        print("Top Label Pairs by Overlap Count:")
        print(overlap_summary.head(10).to_string(index=False))
        
        print(f"\nTotal Analysis Cases Available: {len(analysis_df)}")
        print(f"Unique Label Pairs: {len(overlap_summary)}")
        print(f"Providers with Overlaps: {analysis_df['outlier_pin'].nunique()}")
        
        # Show distribution of overlap scores
        print(f"\nOverlap Score Distribution:")
        print(f"  Mean: {analysis_df['overlap_score'].mean():.3f}")
        print(f"  Median: {analysis_df['overlap_score'].median():.3f}")
        print(f"  Max: {analysis_df['overlap_score'].max():.3f}")
        print(f"  Min: {analysis_df['overlap_score'].min():.3f}")
    
    return summary_output

summary_widget = create_summary_widget()

# Step 7: Create the main interface layout
print("\nStep 3: Creating main interface...")

main_interface = widgets.VBox([
    widgets.HTML("<h3>🔬 Embedding Overlap Analysis Interface</h3>"),
    summary_widget,
    widgets.HTML("<br><h4>Select Analysis Case:</h4>"),
    analysis_dropdown,
    widgets.HTML("<br>"),
    info_output,
    analysis_output
])

print("✅ Interactive widget created successfully!")
print("\nTo use the interface:")
print("1. Review the summary statistics above")
print("2. Select an analysis case from the dropdown")
print("3. View the detailed comparison results")

# Display the interface
display(main_interface)

print(f"\n📋 Interface Ready:")
print(f"   - {len(dropdown_options)} analysis cases available")
print(f"   - Dropdown populated with readable descriptions")
print(f"   - Selection handler connected")
print(f"   - Ready for Stage 2B (detailed analysis functions)")