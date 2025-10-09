import pandas as pd
import numpy as np
import datetime
import gc

# Configuration
EXPORT_LIMIT = 5  # Change to None for all providers
EXPORT_FILENAME = f"Overlapping_Provider_Analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

print(f"Excel Export Configuration:")
print(f"Export Limit: {EXPORT_LIMIT if EXPORT_LIMIT else 'All providers'}")
print(f"Output File: {EXPORT_FILENAME}")

def clean_tab_name(name):
    """Clean provider name for Excel tab naming."""
    invalid_chars = ['\\', '/', '*', '[', ']', ':', '?', '<', '>', '|']
    clean_name = str(name)
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '_')
    return clean_name[:25]

def prepare_export_data():
    """Prepare all data for export."""
    
    # Get top overlap cases
    export_cases = analysis_df.nlargest(EXPORT_LIMIT if EXPORT_LIMIT else len(analysis_df), 'overlap_score')
    
    # Create summary statistics
    summary_stats = overlap_summary.copy()
    
    # Create navigation index
    navigation_index = []
    for idx, (_, row) in enumerate(export_cases.iterrows(), 1):
        clean_name = clean_tab_name(row['outlier_name'])
        navigation_index.append({
            'Case_Number': idx,
            'Provider_Name': row['outlier_name'],
            'Label_A': row['outlier_label'].title(),
            'Label_B': row['overlap_label'].title(), 
            'Overlap_Score': round(row['overlap_score'], 3),
            'Tab_Name': f"Case_{idx}_{clean_name}"
        })
    
    navigation_df = pd.DataFrame(navigation_index)
    
    print(f"Prepared {len(export_cases)} cases for export")
    return export_cases, summary_stats, navigation_df

# Execute Step 1
export_cases, summary_stats, navigation_df = prepare_export_data()
print("✅ Step 1 Complete: Data prepared")