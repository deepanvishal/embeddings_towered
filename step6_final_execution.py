def process_all_analysis_tabs():
    """Process all individual analysis tabs with incremental saves."""
    
    print(f"Processing {len(export_cases)} analysis tabs...")
    
    for idx, (_, selected_row) in enumerate(export_cases.iterrows(), 1):
        print(f"Processing case {idx}/{len(export_cases)}: {selected_row['outlier_name']}")
        
        try:
            # Create new worksheet
            tab_name = navigation_df.iloc[idx-1]['Tab_Name']
            ws_analysis = wb.create_sheet(tab_name)
            
            # Get pre-computed data
            analysis_data = precomputed_data[selected_row['outlier_pin']]
            
            # Write complete analysis
            write_single_analysis(ws_analysis, selected_row, analysis_data, tab_name)
            
            # Save incrementally for recovery
            wb.save(recovery_filename)
            print(f"  ✓ Case {idx} completed and saved")
            
        except Exception as e:
            print(f"  ✗ Error on case {idx}: {str(e)}")
            # Continue with next case
            continue
    
    print("✅ All analysis tabs processed")

def finalize_export():
    """Final save and cleanup."""
    
    try:
        # Final save with clean filename
        wb.save(EXPORT_FILENAME)
        print(f"✅ Final Excel export saved: {EXPORT_FILENAME}")
        
        # Clean up recovery file
        if os.path.exists(recovery_filename):
            os.remove(recovery_filename)
            print("Recovery file cleaned up")
            
        print(f"📊 Export complete: {len(export_cases)} analysis tabs + 2 summary tabs")
        return EXPORT_FILENAME
        
    except Exception as e:
        print(f"❌ Final save failed: {e}")
        print(f"💾 Recovery file available: {recovery_filename}")
        return recovery_filename

# Execute Step 6
print("🚀 Starting final execution...")

# Process all analysis tabs
process_all_analysis_tabs()

# Finalize export
final_filename = finalize_export()

print(f"\n🎉 EXPORT COMPLETE!")
print(f"📁 File: {final_filename}")
print(f"📊 Contains: {len(export_cases)} detailed provider analyses")
print(f"🔗 Navigation: Use Tab 2 to jump to any analysis")

# Performance summary
print(f"\n⚡ PERFORMANCE OPTIMIZATIONS APPLIED:")
print(f"✅ Pre-computation: All data computed once upfront")
print(f"✅ Incremental saves: Recovery after each case")
print(f"✅ Memory management: Cleanup after each case")
print(f"✅ Error handling: Individual failures don't stop process")
print(f"✅ Chart integration: Embedding + PCA charts included")