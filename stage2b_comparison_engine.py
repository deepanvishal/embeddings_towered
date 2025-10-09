import pandas as pd
import numpy as np

print("Stage 2B: Detailed Comparison Engine")
print("=" * 40)

# Step 1: Procedure analysis functions
def get_provider_procedures(pin, top_n=10):
    """Get top procedures for a provider"""
    procedures = amt_smry_df[
        (amt_smry_df['PIN'] == pin) & 
        (amt_smry_df['code_type'] == 'Procedure')
    ].copy()
    
    if len(procedures) == 0:
        return pd.DataFrame(columns=['code', 'code_desc', 'claims', 'pct'])
    
    # Group by procedure and sum claims
    proc_summary = procedures.groupby(['code', 'code_desc'])['claims'].sum().reset_index()
    proc_summary['pct'] = proc_summary['claims'] / proc_summary['claims'].sum() * 100
    
    return proc_summary.nlargest(top_n, 'claims').round(2)

def compare_procedures(pin_a, pin_b, provider_a_name, provider_b_name):
    """Compare procedures between two providers"""
    proc_a = get_provider_procedures(pin_a)
    proc_b = get_provider_procedures(pin_b)
    
    # Find common procedures
    common = proc_a.merge(proc_b, on=['code', 'code_desc'], suffixes=('_a', '_b'), how='inner')
    
    if len(common) > 0:
        common['claims_diff'] = common['claims_a'] - common['claims_b']
        common['pct_diff'] = common['pct_a'] - common['pct_b']
        common = common.sort_values('claims_a', ascending=False)
    
    return {
        'provider_a': proc_a,
        'provider_b': proc_b,
        'common': common,
        'total_a': proc_a['claims'].sum() if len(proc_a) > 0 else 0,
        'total_b': proc_b['claims'].sum() if len(proc_b) > 0 else 0,
        'overlap_procedures': len(common)
    }

# Step 2: Demographics comparison functions
def get_provider_demographics(pin):
    """Get demographics for a provider"""
    demo_cols = ['peds_pct', 'adults_pct', 'seniors_pct', 'Female_pct', 
                 'Inpatient_pct', 'Emergency_pct']
    
    provider_demo = member_df[member_df['PIN'] == pin]
    if len(provider_demo) == 0:
        return pd.Series(index=demo_cols, dtype=float)
    
    return provider_demo[demo_cols].iloc[0]

def compare_demographics(pin_a, pin_b, provider_a_name, provider_b_name):
    """Compare demographics between two providers"""
    demo_a = get_provider_demographics(pin_a)
    demo_b = get_provider_demographics(pin_b)
    
    comparison = pd.DataFrame({
        'metric': demo_a.index,
        provider_a_name: demo_a.values,
        provider_b_name: demo_b.values
    })
    
    comparison['difference'] = comparison[provider_a_name] - comparison[provider_b_name]
    comparison['abs_difference'] = abs(comparison['difference'])
    
    return comparison.round(2)

# Step 3: Medical cost comparison functions
def get_provider_costs(pin):
    """Get cost distribution for a provider"""
    cost_cols = [col for col in member_df.columns if col.startswith('med_cost_ctg_cd_')]
    
    provider_costs = member_df[member_df['PIN'] == pin]
    if len(provider_costs) == 0:
        return pd.Series(index=cost_cols, dtype=float)
    
    return provider_costs[cost_cols].iloc[0]

def compare_costs(pin_a, pin_b, provider_a_name, provider_b_name):
    """Compare cost distributions between two providers"""
    costs_a = get_provider_costs(pin_a)
    costs_b = get_provider_costs(pin_b)
    
    comparison = pd.DataFrame({
        'cost_category': costs_a.index,
        provider_a_name: costs_a.values,
        provider_b_name: costs_b.values
    })
    
    comparison['difference'] = comparison[provider_a_name] - comparison[provider_b_name]
    comparison['abs_difference'] = abs(comparison['difference'])
    
    return comparison.round(2)

# Step 4: Enhanced analysis function
def run_detailed_analysis_enhanced(selected_row):
    """Enhanced detailed analysis with actual data"""
    
    # Extract provider information
    outlier_pin = selected_row['outlier_pin']
    outlier_name = selected_row['outlier_name']
    outlier_label = selected_row['outlier_label']
    
    typical_same_pin = selected_row['typical_same_pin']
    typical_same_name = selected_row['typical_same_name']
    
    typical_other_pin = selected_row['typical_other_pin']
    typical_other_name = selected_row['typical_other_name']
    overlap_label = selected_row['overlap_label']
    
    print(f"🔍 DETAILED ANALYSIS: {outlier_name}")
    print("=" * 80)
    
    # PART 1: Within-Group Analysis
    print(f"\n📊 PART 1: WITHIN-GROUP ANALYSIS ({outlier_label} GROUP)")
    print(f"Comparing: {outlier_name} vs {typical_same_name}")
    print("-" * 60)
    
    # Procedure comparison A vs B
    print(f"\n🏥 PROCEDURE COMPARISON:")
    proc_comp1 = compare_procedures(outlier_pin, typical_same_pin, outlier_name, typical_same_name)
    
    print(f"\n{outlier_name} - Top 10 Procedures:")
    if len(proc_comp1['provider_a']) > 0:
        display(proc_comp1['provider_a'][['code_desc', 'claims', 'pct']].head(10))
        print(f"Total procedures: {proc_comp1['total_a']:,}")
    else:
        print("No procedure data available")
    
    print(f"\n{typical_same_name} - Top 10 Procedures:")
    if len(proc_comp1['provider_b']) > 0:
        display(proc_comp1['provider_b'][['code_desc', 'claims', 'pct']].head(10))
        print(f"Total procedures: {proc_comp1['total_b']:,}")
    else:
        print("No procedure data available")
    
    if len(proc_comp1['common']) > 0:
        print(f"\nCommon Procedures ({len(proc_comp1['common'])} found):")
        display(proc_comp1['common'][['code_desc', 'claims_a', 'claims_b', 'pct_a', 'pct_b']].head())
    else:
        print("\nNo common procedures in top 10")
    
    # Demographics comparison A vs B
    print(f"\n👥 DEMOGRAPHICS COMPARISON:")
    demo_comp1 = compare_demographics(outlier_pin, typical_same_pin, outlier_name, typical_same_name)
    display(demo_comp1)
    
    # Cost comparison A vs B
    print(f"\n💰 MEDICAL COST COMPARISON:")
    cost_comp1 = compare_costs(outlier_pin, typical_same_pin, outlier_name, typical_same_name)
    display(cost_comp1)
    
    # PART 2: Cross-Group Analysis
    print(f"\n📊 PART 2: CROSS-GROUP ANALYSIS ({outlier_label} vs {overlap_label})")
    print(f"Comparing: {outlier_name} vs {typical_other_name}")
    print("-" * 60)
    
    # Procedure comparison A vs C
    print(f"\n🏥 PROCEDURE COMPARISON:")
    proc_comp2 = compare_procedures(outlier_pin, typical_other_pin, outlier_name, typical_other_name)
    
    print(f"\n{outlier_name} - Top 10 Procedures:")
    if len(proc_comp2['provider_a']) > 0:
        display(proc_comp2['provider_a'][['code_desc', 'claims', 'pct']].head(10))
    else:
        print("No procedure data available")
    
    print(f"\n{typical_other_name} - Top 10 Procedures:")
    if len(proc_comp2['provider_b']) > 0:
        display(proc_comp2['provider_b'][['code_desc', 'claims', 'pct']].head(10))
        print(f"Total procedures: {proc_comp2['total_b']:,}")
    else:
        print("No procedure data available")
    
    if len(proc_comp2['common']) > 0:
        print(f"\nCommon Procedures ({len(proc_comp2['common'])} found):")
        display(proc_comp2['common'][['code_desc', 'claims_a', 'claims_b', 'pct_a', 'pct_b']].head())
    else:
        print("\nNo common procedures in top 10")
    
    # Demographics comparison A vs C
    print(f"\n👥 DEMOGRAPHICS COMPARISON:")
    demo_comp2 = compare_demographics(outlier_pin, typical_other_pin, outlier_name, typical_other_name)
    display(demo_comp2)
    
    # Cost comparison A vs C
    print(f"\n💰 MEDICAL COST COMPARISON:")
    cost_comp2 = compare_costs(outlier_pin, typical_other_pin, outlier_name, typical_other_name)
    display(cost_comp2)
    
    # SUMMARY INSIGHTS
    print(f"\n📋 SUMMARY INSIGHTS:")
    print("=" * 40)
    print(f"Distance Analysis:")
    print(f"  • {outlier_name} to {outlier_label} centroid: {selected_row['own_distance']:.4f}")
    print(f"  • {outlier_name} to {overlap_label} centroid: {selected_row['other_distance']:.4f}")
    print(f"  • Overlap score: {selected_row['overlap_score']:.4f}")
    
    # Calculate similarity insights
    demo_sim_same = (demo_comp1['abs_difference'] < 5).sum()
    demo_sim_other = (demo_comp2['abs_difference'] < 5).sum() 
    
    print(f"\nSimilarity Analysis:")
    print(f"  • Demographics similar to {typical_same_name}: {demo_sim_same}/6 metrics")
    print(f"  • Demographics similar to {typical_other_name}: {demo_sim_other}/6 metrics")
    print(f"  • Common procedures with {typical_same_name}: {proc_comp1['overlap_procedures']}")
    print(f"  • Common procedures with {typical_other_name}: {proc_comp2['overlap_procedures']}")
    
    return {
        'proc_comp1': proc_comp1,
        'proc_comp2': proc_comp2,
        'demo_comp1': demo_comp1,
        'demo_comp2': demo_comp2,
        'cost_comp1': cost_comp1,
        'cost_comp2': cost_comp2
    }

# Step 5: Update the widget's analysis function
def update_analysis_function():
    """Replace the placeholder analysis function"""
    global run_detailed_analysis
    run_detailed_analysis = run_detailed_analysis_enhanced
    print("✅ Analysis function updated in widget")

print("✅ Comparison engine functions created:")
print("  - get_provider_procedures()")
print("  - compare_procedures()")
print("  - get_provider_demographics()")
print("  - compare_demographics()")
print("  - get_provider_costs()")
print("  - compare_costs()")
print("  - run_detailed_analysis_enhanced()")

print("\n🔄 Updating widget analysis function...")
update_analysis_function()

print("\n✅ Stage 2B Complete!")
print("The interactive widget now has full comparison capabilities.")
print("Select an analysis case from the dropdown to see detailed comparisons.")