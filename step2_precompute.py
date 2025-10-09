def precompute_all_analysis_data():
    """Pre-compute all analysis data for performance."""
    
    print("Pre-computing analysis data for all cases...")
    precomputed_data = {}
    
    for idx, (_, row) in enumerate(export_cases.iterrows(), 1):
        print(f"Pre-computing case {idx}/{len(export_cases)}: {row['outlier_name']}")
        
        pin_outlier = row['outlier_pin']
        pin_same = row['typical_same_pin']
        pin_other = row['typical_other_pin']
        
        # Get all provider data
        case_data = {
            'procedures': {
                'outlier': get_provider_codes(pin_outlier, 'Procedure', 10),
                'typical_same': get_provider_codes(pin_same, 'Procedure', 10),
                'typical_other': get_provider_codes(pin_other, 'Procedure', 10),
                'common_same': find_common_codes(pin_outlier, pin_same, 'Procedure', 10),
                'common_other': find_common_codes(pin_outlier, pin_other, 'Procedure', 10)
            },
            'diagnoses': {
                'outlier': get_provider_codes(pin_outlier, 'Diagnosis', 10),
                'typical_same': get_provider_codes(pin_same, 'Diagnosis', 10),
                'typical_other': get_provider_codes(pin_other, 'Diagnosis', 10),
                'common_same': find_common_codes(pin_outlier, pin_same, 'Diagnosis', 10),
                'common_other': find_common_codes(pin_outlier, pin_other, 'Diagnosis', 10)
            },
            'demographics': {
                'outlier': get_provider_demographics(pin_outlier),
                'typical_same': get_provider_demographics(pin_same),
                'typical_other': get_provider_demographics(pin_other)
            },
            'costs': {
                'outlier': get_provider_costs(pin_outlier),
                'typical_same': get_provider_costs(pin_same),
                'typical_other': get_provider_costs(pin_other)
            }
        }
        
        # Pre-compute comparison tables
        case_data['demographics']['comparison_same'] = create_comparison_table(
            case_data['demographics']['outlier'], case_data['demographics']['typical_same'],
            row['outlier_name'], row['typical_same_name'], 'demo'
        )
        case_data['demographics']['comparison_other'] = create_comparison_table(
            case_data['demographics']['outlier'], case_data['demographics']['typical_other'],
            row['outlier_name'], row['typical_other_name'], 'demo'
        )
        case_data['costs']['comparison_same'] = create_comparison_table(
            case_data['costs']['outlier'], case_data['costs']['typical_same'],
            row['outlier_name'], row['typical_same_name'], 'cost'
        )
        case_data['costs']['comparison_other'] = create_comparison_table(
            case_data['costs']['outlier'], case_data['costs']['typical_other'],
            row['outlier_name'], row['typical_other_name'], 'cost'
        )
        
        precomputed_data[pin_outlier] = case_data
        
        # Clean up memory
        gc.collect()
    
    print("✅ Pre-computation complete!")
    return precomputed_data

# Execute Step 2
precomputed_data = precompute_all_analysis_data()
print("✅ Step 2 Complete: All analysis data pre-computed")