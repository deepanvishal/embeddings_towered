import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_code_frequencies(df, code_col='code', 
                             code_type_col=None,
                             count_col='frequency',
                             code_desc_col=None):
    """
    Analyze code frequencies using IQR method to identify low-frequency outliers
    Shows distribution ranges and counts for codes (procedures, diagnoses, etc.)
    
    Args:
        df: DataFrame with codes and their frequencies
            - code_col: Column containing the code (e.g., procedure code, diagnosis code)
            - code_type_col: Optional column to specify code type (e.g., 'procedure', 'diagnosis')
            - count_col: Column containing frequency/count for each code
            - code_desc_col: Optional column with code descriptions
    
    Returns:
        Dictionary with thresholds, low-frequency codes, and distribution analysis
    """
    
    # Prepare data
    print("Analyzing code frequency distribution...")
    code_stats = df.copy()
    
    # Rename columns for consistency
    col_mapping = {code_col: 'code', count_col: 'frequency'}
    if code_type_col:
        col_mapping[code_type_col] = 'code_type'
    if code_desc_col:
        col_mapping[code_desc_col] = 'description'
    
    code_stats = code_stats.rename(columns=col_mapping)
    
    print(f"\nTotal unique codes: {len(code_stats):,}")
    print(f"Total occurrences: {code_stats['frequency'].sum():,}")
    print(f"Mean frequency per code: {code_stats['frequency'].mean():.2f}")
    print(f"Median frequency per code: {code_stats['frequency'].median():.2f}")
    
    # Calculate IQR statistics
    Q1 = code_stats['frequency'].quantile(0.25)
    Q2 = code_stats['frequency'].quantile(0.50)  # Median
    Q3 = code_stats['frequency'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate outlier thresholds
    # For LOW frequency outliers, we use Q1 - 1.5*IQR
    lower_outlier = max(1, Q1 - 1.5 * IQR)  # At least 1
    # For HIGH frequency outliers, we use Q3 + 1.5*IQR
    upper_outlier = Q3 + 1.5 * IQR
    
    thresholds = {
        'min': code_stats['frequency'].min(),
        'Q1': Q1,
        'Q2': Q2,
        'Q3': Q3,
        'max': code_stats['frequency'].max(),
        'IQR': IQR,
        'lower_outlier': lower_outlier,
        'upper_outlier': upper_outlier
    }
    
    print("\n" + "="*70)
    print("IQR ANALYSIS - CODE FREQUENCY THRESHOLDS")
    print("="*70)
    print(f"  Min frequency: {thresholds['min']:.1f}")
    print(f"  Q1 (25th percentile): {Q1:.1f}")
    print(f"  Median (50th percentile): {Q2:.1f}")
    print(f"  Q3 (75th percentile): {Q3:.1f}")
    print(f"  Max frequency: {thresholds['max']:.1f}")
    print(f"  IQR: {IQR:.1f}")
    print(f"  Lower outlier threshold (Q1-1.5*IQR): {lower_outlier:.1f}")
    print(f"  Upper outlier threshold (Q3+1.5*IQR): {upper_outlier:.1f}")
    
    # Identify low-frequency codes
    low_freq_codes = code_stats[code_stats['frequency'] < lower_outlier].copy()
    low_freq_codes = low_freq_codes.sort_values('frequency', ascending=True)
    
    print("\n" + "="*70)
    print("LOW FREQUENCY OUTLIER CODES")
    print("="*70)
    print(f"Codes with frequency < {lower_outlier:.1f}: {len(low_freq_codes):,} codes")
    print(f"Percentage of all codes: {len(low_freq_codes)/len(code_stats)*100:.2f}%")
    print(f"Total occurrences from these codes: {low_freq_codes['frequency'].sum():,}")
    print(f"Percentage of all occurrences: {low_freq_codes['frequency'].sum()/code_stats['frequency'].sum()*100:.2f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Code Frequency Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    data = code_stats['frequency']
    
    # Row 1, Column 1: Original distribution
    ax1 = axes[0, 0]
    ax1.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.1f}')
    ax1.axvline(Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.1f}')
    ax1.axvline(Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.1f}')
    ax1.axvline(lower_outlier, color='darkred', linestyle=':', linewidth=2, 
               label=f'Low: {lower_outlier:.1f}')
    ax1.axvline(upper_outlier, color='purple', linestyle=':', linewidth=2, 
               label=f'High: {upper_outlier:.1f}')
    ax1.set_xlabel('Code Frequency', fontsize=11)
    ax1.set_ylabel('Number of Codes', fontsize=11)
    ax1.set_title('Original Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Row 1, Column 2: Log-transformed distribution
    ax2 = axes[0, 1]
    log_data = np.log10(data + 1)
    log_Q1 = np.log10(Q1 + 1)
    log_Q2 = np.log10(Q2 + 1)
    log_Q3 = np.log10(Q3 + 1)
    log_lower = np.log10(lower_outlier + 1)
    log_upper = np.log10(upper_outlier + 1)
    
    ax2.hist(log_data, bins=50, color='seagreen', alpha=0.7, edgecolor='black')
    ax2.axvline(log_Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.1f}')
    ax2.axvline(log_Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.1f}')
    ax2.axvline(log_Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.1f}')
    ax2.axvline(log_lower, color='darkred', linestyle=':', linewidth=2, 
               label=f'Low: {lower_outlier:.1f}')
    ax2.axvline(log_upper, color='purple', linestyle=':', linewidth=2, 
               label=f'High: {upper_outlier:.1f}')
    ax2.set_xlabel('Log10(Frequency + 1)', fontsize=11)
    ax2.set_ylabel('Number of Codes', fontsize=11)
    ax2.set_title('Log-Transformed Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Row 1, Column 3: Box plot
    ax3 = axes[0, 2]
    box_data = ax3.boxplot([data], vert=True, patch_artist=True, widths=0.6)
    for patch in box_data['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax3.axhline(Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1:.1f}')
    ax3.axhline(Q2, color='green', linestyle='--', linewidth=2, label=f'Median: {Q2:.1f}')
    ax3.axhline(Q3, color='red', linestyle='--', linewidth=2, label=f'Q3: {Q3:.1f}')
    ax3.axhline(lower_outlier, color='darkred', linestyle=':', linewidth=2, 
               label=f'Low: {lower_outlier:.1f}')
    ax3.axhline(upper_outlier, color='purple', linestyle=':', linewidth=2, 
               label=f'High: {upper_outlier:.1f}')
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Box Plot', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticklabels([''])
    
    # Row 2, Column 1: Low frequency codes histogram (zoomed)
    ax4 = axes[1, 0]
    if len(low_freq_codes) > 0:
        ax4.hist(low_freq_codes['frequency'], bins=30, color='crimson', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Code Frequency', fontsize=11)
        ax4.set_ylabel('Number of Codes', fontsize=11)
        ax4.set_title(f'Low-Frequency Outliers (< {lower_outlier:.1f})', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No low-frequency outliers', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # Row 2, Column 2: Top 20 lowest frequency codes
    ax5 = axes[1, 1]
    if len(low_freq_codes) > 0:
        top_20_low = low_freq_codes.head(20)
        y_pos = np.arange(len(top_20_low))
        ax5.barh(y_pos, top_20_low['frequency'], color='crimson', alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top_20_low['code'].astype(str), fontsize=8)
        ax5.set_xlabel('Frequency', fontsize=11)
        ax5.set_title('Top 20 Lowest Frequency Codes', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
    else:
        ax5.text(0.5, 0.5, 'No low-frequency outliers', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # Row 2, Column 3: Distribution ranges
    ax6 = axes[1, 2]
    
    # Count codes in each range
    below_lower = (code_stats['frequency'] < lower_outlier).sum()
    lower_to_q1 = ((code_stats['frequency'] >= lower_outlier) & (code_stats['frequency'] < Q1)).sum()
    q1_to_q2 = ((code_stats['frequency'] >= Q1) & (code_stats['frequency'] < Q2)).sum()
    q2_to_q3 = ((code_stats['frequency'] >= Q2) & (code_stats['frequency'] < Q3)).sum()
    q3_to_upper = ((code_stats['frequency'] >= Q3) & (code_stats['frequency'] < upper_outlier)).sum()
    above_upper = (code_stats['frequency'] >= upper_outlier).sum()
    
    categories = ['Very Low\n(Outliers)', 'Low', 'Below Median', 'Above Median', 'High', 'Very High\n(Outliers)']
    counts = [below_lower, lower_to_q1, q1_to_q2, q2_to_q3, q3_to_upper, above_upper]
    colors_bar = ['darkred', 'orange', 'yellow', 'lightgreen', 'green', 'purple']
    
    bars = ax6.bar(range(len(categories)), counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(categories)))
    ax6.set_xticklabels(categories, fontsize=9, rotation=45, ha='right')
    ax6.set_ylabel('Number of Codes', fontsize=11)
    ax6.set_title('Code Distribution by Range', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/len(code_stats)*100:.1f}%)',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution analysis
    print("\n" + "="*70)
    print("CODE DISTRIBUTION ACROSS FREQUENCY RANGES")
    print("="*70)
    
    total = len(code_stats)
    
    print(f"\n  Below lower outlier (<{lower_outlier:.1f}): {below_lower:,} codes ({below_lower/total*100:.1f}%)")
    print(f"  Lower outlier to Q1 ({lower_outlier:.1f} to {Q1:.1f}): {lower_to_q1:,} codes ({lower_to_q1/total*100:.1f}%)")
    print(f"  Q1 to Median ({Q1:.1f} to {Q2:.1f}): {q1_to_q2:,} codes ({q1_to_q2/total*100:.1f}%)")
    print(f"  Median to Q3 ({Q2:.1f} to {Q3:.1f}): {q2_to_q3:,} codes ({q2_to_q3/total*100:.1f}%)")
    print(f"  Q3 to upper outlier ({Q3:.1f} to {upper_outlier:.1f}): {q3_to_upper:,} codes ({q3_to_upper/total*100:.1f}%)")
    print(f"  Above upper outlier (≥{upper_outlier:.1f}): {above_upper:,} codes ({above_upper/total*100:.1f}%)")
    
    # Show sample of low-frequency codes
    if len(low_freq_codes) > 0:
        print("\n" + "="*70)
        print("SAMPLE OF LOW FREQUENCY OUTLIER CODES (Bottom 10)")
        print("="*70)
        
        display_cols = ['code', 'frequency']
        if 'description' in low_freq_codes.columns:
            display_cols.append('description')
        if 'code_type' in low_freq_codes.columns:
            display_cols.insert(1, 'code_type')
        
        print(low_freq_codes[display_cols].head(10).to_string(index=False))
    
    return {
        'code_stats': code_stats,
        'thresholds': thresholds,
        'low_frequency_codes': low_freq_codes,
        'distribution_counts': {
            'below_lower_outlier': below_lower,
            'lower_to_q1': lower_to_q1,
            'q1_to_median': q1_to_q2,
            'median_to_q3': q2_to_q3,
            'q3_to_upper': q3_to_upper,
            'above_upper_outlier': above_upper
        }
    }


def analyze_codes_by_type(df, code_col='code', 
                          code_type_col='code_type',
                          count_col='frequency',
                          code_desc_col=None):
    """
    Analyze code frequencies separately for each code type (e.g., procedures vs diagnoses)
    
    Args:
        df: DataFrame with codes, their type, and frequencies
        code_col: Column containing the code
        code_type_col: Column specifying code type
        count_col: Column containing frequency/count
        code_desc_col: Optional column with code descriptions
    
    Returns:
        Dictionary with results for each code type
    """
    
    results_by_type = {}
    code_types = df[code_type_col].unique()
    
    print(f"\nAnalyzing {len(code_types)} code types: {', '.join(code_types)}")
    print("="*70)
    
    for code_type in code_types:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {code_type.upper()}")
        print(f"{'='*70}")
        
        type_df = df[df[code_type_col] == code_type].copy()
        
        results = analyze_code_frequencies(
            type_df, 
            code_col=code_col,
            count_col=count_col,
            code_desc_col=code_desc_col
        )
        
        results_by_type[code_type] = results
    
    return results_by_type


# Usage examples:

# Example 1: Simple code frequency analysis
# df = pd.DataFrame({
#     'procedure_code': ['99213', '99214', '70450', ...],
#     'count': [1234, 5678, 2, ...]
# })
# results = analyze_code_frequencies(df, code_col='procedure_code', count_col='count')

# Example 2: With code descriptions
# df = pd.DataFrame({
#     'code': ['99213', '99214', '70450', ...],
#     'description': ['Office visit', 'Office visit', 'CT scan', ...],
#     'frequency': [1234, 5678, 2, ...]
# })
# results = analyze_code_frequencies(df, code_col='code', count_col='frequency', 
#                                    code_desc_col='description')

# Example 3: Separate analysis by code type
# df = pd.DataFrame({
#     'code': ['99213', 'A1234', '70450', 'B5678', ...],
#     'code_type': ['procedure', 'diagnosis', 'procedure', 'diagnosis', ...],
#     'frequency': [1234, 5678, 2, 890, ...]
# })
# results = analyze_codes_by_type(df, code_col='code', code_type_col='code_type', 
#                                  count_col='frequency')
