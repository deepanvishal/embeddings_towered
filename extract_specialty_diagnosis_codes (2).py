# ============================================================================
# EXTRACT SPECIALTY-SPECIFIC TOP DIAGNOSIS CODES
# ============================================================================
# Run this after preprocessing to get diagnosis code mappings

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz

print("\n" + "="*80)
print("EXTRACTING SPECIALTY-SPECIFIC TOP DIAGNOSIS CODES")
print("="*80)

# Load data
print("\nLoading data...")
diag_matrix = load_npz('diagnosis_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

# Load diagnosis code mapping (from preprocessing)
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
diagnosis_codes = sorted(diagnosis_df['code'].unique())
diag_to_idx = {code: idx for idx, code in enumerate(diagnosis_codes)}

print(f"Diagnosis matrix: {diag_matrix.shape}")
print(f"Total PINs: {len(all_pins)}")
print(f"Labeled PINs: {len(pin_to_label)}")
print(f"Total diagnosis codes: {len(diagnosis_codes)}")

# Create PIN to index mapping
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# Parameters
TOP_K_CODES = 200  # Number of top codes per specialty

# Get unique specialties
unique_specialties = sorted(set(pin_to_label.values()))
n_specialties = len(unique_specialties)

print(f"\nProcessing {n_specialties} specialties...")
print(f"Extracting top {TOP_K_CODES} diagnosis codes per specialty")

# Initialize storage
specialty_code_indices = {}
specialty_code_names = {}
specialty_stats = {}

for specialty_name in unique_specialties:
    # Find all PINs with this label
    specialty_pins = [pin for pin, label in pin_to_label.items() if label == specialty_name]
    
    if len(specialty_pins) == 0:
        print(f"\nWARNING: No hospitals found for {specialty_name}")
        continue
    
    # Get indices of these PINs in the full PIN list
    specialty_pin_indices = [pin_to_idx[pin] for pin in specialty_pins]
    
    # Extract diagnosis matrix for this specialty
    specialty_diag_matrix = diag_matrix[specialty_pin_indices, :]
    
    # Sum claims across all hospitals in this specialty
    # Shape: (n_codes,)
    code_totals = np.array(specialty_diag_matrix.sum(axis=0)).flatten()
    
    # Get top K codes by total claims
    top_k_indices = np.argsort(code_totals)[::-1][:TOP_K_CODES]
    
    # Filter out codes with zero claims
    top_k_indices = top_k_indices[code_totals[top_k_indices] > 0]
    
    # Get actual code names
    idx_to_diag = {idx: code for code, idx in diag_to_idx.items()}
    top_k_code_names = [idx_to_diag[idx] for idx in top_k_indices]
    
    # Store results
    specialty_code_indices[specialty_name] = top_k_indices.tolist()
    specialty_code_names[specialty_name] = top_k_code_names
    
    # Calculate statistics
    total_claims = code_totals[top_k_indices].sum()
    n_nonzero = len(top_k_indices)
    coverage = total_claims / code_totals.sum() if code_totals.sum() > 0 else 0
    
    specialty_stats[specialty_name] = {
        'n_hospitals': len(specialty_pins),
        'n_codes_extracted': n_nonzero,
        'total_claims': int(total_claims),
        'coverage': coverage
    }
    
    print(f"\n{specialty_name}:")
    print(f"  Hospitals: {len(specialty_pins)}")
    print(f"  Top codes extracted: {n_nonzero}/{TOP_K_CODES}")
    print(f"  Coverage: {coverage:.2%} of total claims")
    print(f"  Top 5 codes: {top_k_code_names[:5]}")

# ============================================================================
# SAVE SPECIALTY DIAGNOSIS CODE MAPPINGS
# ============================================================================
print("\n" + "="*80)
print("SAVING SPECIALTY DIAGNOSIS CODE MAPPINGS")
print("="*80)

specialty_diagnosis_mappings = {
    'code_indices': specialty_code_indices,  # {specialty: [col_indices]}
    'code_names': specialty_code_names,      # {specialty: [code_strings]}
    'stats': specialty_stats,                 # {specialty: {stats}}
    'top_k': TOP_K_CODES,
    'specialties': unique_specialties
}

with open('specialty_diagnosis_mappings.pkl', 'wb') as f:
    pickle.dump(specialty_diagnosis_mappings, f)

print(f"\nSaved: specialty_diagnosis_mappings.pkl")
print(f"  Contains mappings for {len(specialty_code_indices)} specialties")
print(f"  Each specialty has up to {TOP_K_CODES} diagnosis codes")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

total_hospitals = sum(s['n_hospitals'] for s in specialty_stats.values())
avg_coverage = np.mean([s['coverage'] for s in specialty_stats.values()])

print(f"\nTotal labeled hospitals: {total_hospitals}")
print(f"Average code coverage: {avg_coverage:.2%}")
print(f"\nHospitals per specialty:")
for specialty, stats in sorted(specialty_stats.items(), key=lambda x: x[1]['n_hospitals'], reverse=True):
    print(f"  {specialty:20s}: {stats['n_hospitals']:4d} hospitals, {stats['n_codes_extracted']:3d} codes")

print("\n" + "="*80)
print("DIAGNOSIS CODE EXTRACTION COMPLETE")
print("="*80)
print("\nNext: Run phase1_train_diagnosis_autoencoders.py")
