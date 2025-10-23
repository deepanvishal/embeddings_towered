import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

print("\n" + "="*80)
print("PROCESSING LINEAR TOWERS & COMBINING ALL 6 TOWERS")
print("="*80)

# ============================================================================
# LOAD ALL DATA
# ============================================================================

print("\nLoading data...")

# Load procedure embeddings (128 dims)
proc_emb_df = pd.read_parquet('hospital_embeddings_128d.parquet')
proc_emb_cols = [col for col in proc_emb_df.columns if col.startswith('emb_')]
proc_embeddings = proc_emb_df[proc_emb_cols].values

# Load diagnosis embeddings (128 dims)
diag_emb_df = pd.read_parquet('hospital_diagnosis_embeddings_128d.parquet')
diag_emb_cols = [col for col in diag_emb_df.columns if col.startswith('emb_')]
diag_embeddings = diag_emb_df[diag_emb_cols].values

# Load member vectors (linear towers)
member_matrix = np.load('member_vectors.npy')

# Load metadata to understand member_matrix structure
with open('vectors_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load PINs
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

print(f"\nLoaded data:")
print(f"  Procedure embeddings: {proc_embeddings.shape}")
print(f"  Diagnosis embeddings: {diag_embeddings.shape}")
print(f"  Member matrix: {member_matrix.shape}")
print(f"  Total PINs: {len(all_pins)}")
print(f"\nMetadata:")
print(f"  n_member columns: {metadata['n_member']}")

# ============================================================================
# SPLIT MEMBER MATRIX INTO 4 LINEAR TOWERS
# ============================================================================
# NOTE: Adjust these indices based on your actual member_df column structure
# This assumes: [demographics (5), place (4), cost (11), PIN (2)]
# IMPORTANT: Verify this matches your actual data!

print("\n" + "="*80)
print("SPLITTING MEMBER MATRIX INTO LINEAR TOWERS")
print("="*80)

# You may need to adjust these based on your member_df structure
demo_start, demo_end = 0, 5
place_start, place_end = 5, 9
cost_start, cost_end = 9, 20
pin_start, pin_end = 20, 22

demographics = member_matrix[:, demo_start:demo_end]
place = member_matrix[:, place_start:place_end]
cost = member_matrix[:, cost_start:cost_end]
pin_summary = member_matrix[:, pin_start:pin_end]

print(f"\nSplit member matrix:")
print(f"  Demographics: {demographics.shape} (columns {demo_start}:{demo_end})")
print(f"  Place: {place.shape} (columns {place_start}:{place_end})")
print(f"  Cost: {cost.shape} (columns {cost_start}:{cost_end})")
print(f"  PIN summary: {pin_summary.shape} (columns {pin_start}:{pin_end})")

# Verify total
total_linear = demographics.shape[1] + place.shape[1] + cost.shape[1] + pin_summary.shape[1]
print(f"  Total linear dims: {total_linear}")

# ============================================================================
# NORMALIZE LINEAR TOWERS
# ============================================================================

print("\n" + "="*80)
print("NORMALIZING LINEAR TOWERS")
print("="*80)

scaler_demo = StandardScaler()
scaler_place = StandardScaler()
scaler_cost = StandardScaler()
scaler_pin = StandardScaler()

demographics_norm = scaler_demo.fit_transform(demographics)
place_norm = scaler_place.fit_transform(place)
cost_norm = scaler_cost.fit_transform(cost)
pin_norm = scaler_pin.fit_transform(pin_summary)

print(f"\nNormalized linear towers:")
print(f"  Demographics: mean={demographics_norm.mean():.4f}, std={demographics_norm.std():.4f}")
print(f"  Place: mean={place_norm.mean():.4f}, std={place_norm.std():.4f}")
print(f"  Cost: mean={cost_norm.mean():.4f}, std={cost_norm.std():.4f}")
print(f"  PIN: mean={pin_norm.mean():.4f}, std={pin_norm.std():.4f}")

# Save scalers for future use
with open('linear_towers_scalers.pkl', 'wb') as f:
    pickle.dump({
        'scaler_demo': scaler_demo,
        'scaler_place': scaler_place,
        'scaler_cost': scaler_cost,
        'scaler_pin': scaler_pin
    }, f)
print("\n✓ Saved scalers: linear_towers_scalers.pkl")

# ============================================================================
# COMBINE ALL 6 TOWERS
# ============================================================================

print("\n" + "="*80)
print("COMBINING ALL 6 TOWERS")
print("="*80)

# Concatenate all towers
all_embeddings = np.hstack([
    proc_embeddings,      # Tower 1: 128 dims
    diag_embeddings,      # Tower 2: 128 dims
    demographics_norm,    # Tower 3: 5 dims
    place_norm,           # Tower 4: 4 dims
    cost_norm,            # Tower 5: 11 dims
    pin_norm              # Tower 6: 2 dims
])

print(f"\nCombined embeddings shape: {all_embeddings.shape}")
print(f"  Tower 1 (Procedures):  128 dims")
print(f"  Tower 2 (Diagnoses):   128 dims")
print(f"  Tower 3 (Demographics): {demographics_norm.shape[1]} dims")
print(f"  Tower 4 (Place):        {place_norm.shape[1]} dims")
print(f"  Tower 5 (Cost):         {cost_norm.shape[1]} dims")
print(f"  Tower 6 (PIN):          {pin_norm.shape[1]} dims")
print(f"  Total:                  {all_embeddings.shape[1]} dims")

# ============================================================================
# CREATE FINAL DATAFRAME WITH TOWER NAMING
# ============================================================================

print("\n" + "="*80)
print("CREATING FINAL DATAFRAME")
print("="*80)

embedding_data = {'PIN': all_pins}

# Tower 1: Procedures (128 dims)
for i in range(proc_embeddings.shape[1]):
    embedding_data[f'tower1_proc_emb_{i}'] = proc_embeddings[:, i]

# Tower 2: Diagnoses (128 dims)
for i in range(diag_embeddings.shape[1]):
    embedding_data[f'tower2_diag_emb_{i}'] = diag_embeddings[:, i]

# Tower 3: Demographics
for i in range(demographics_norm.shape[1]):
    embedding_data[f'tower3_demo_emb_{i}'] = demographics_norm[:, i]

# Tower 4: Place
for i in range(place_norm.shape[1]):
    embedding_data[f'tower4_plc_emb_{i}'] = place_norm[:, i]

# Tower 5: Cost
for i in range(cost_norm.shape[1]):
    embedding_data[f'tower5_cost_emb_{i}'] = cost_norm[:, i]

# Tower 6: PIN
for i in range(pin_norm.shape[1]):
    embedding_data[f'tower6_pin_emb_{i}'] = pin_norm[:, i]

final_df = pd.DataFrame(embedding_data)

print(f"\nFinal DataFrame shape: {final_df.shape}")
print(f"  Columns: PIN + {final_df.shape[1]-1} embedding dimensions")

# Save
final_df.to_parquet('final_all_towers_278d.parquet', index=False)
np.save('final_all_towers_278d.npy', all_embeddings)

print(f"\n✓ Saved: final_all_towers_278d.parquet")
print(f"✓ Saved: final_all_towers_278d.npy")

# ============================================================================
# SAVE METADATA
# ============================================================================

final_metadata = {
    'total_dims': all_embeddings.shape[1],
    'n_hospitals': all_embeddings.shape[0],
    'tower_dims': {
        'tower1_procedures': 128,
        'tower2_diagnoses': 128,
        'tower3_demographics': demographics_norm.shape[1],
        'tower4_place': place_norm.shape[1],
        'tower5_cost': cost_norm.shape[1],
        'tower6_pin': pin_norm.shape[1]
    },
    'tower_ranges': {
        'tower1_procedures': (0, 128),
        'tower2_diagnoses': (128, 256),
        'tower3_demographics': (256, 256 + demographics_norm.shape[1]),
        'tower4_place': (256 + demographics_norm.shape[1], 256 + demographics_norm.shape[1] + place_norm.shape[1]),
        'tower5_cost': (256 + demographics_norm.shape[1] + place_norm.shape[1], 
                       256 + demographics_norm.shape[1] + place_norm.shape[1] + cost_norm.shape[1]),
        'tower6_pin': (256 + demographics_norm.shape[1] + place_norm.shape[1] + cost_norm.shape[1],
                      all_embeddings.shape[1])
    }
}

with open('final_all_towers_metadata.pkl', 'wb') as f:
    pickle.dump(final_metadata, f)

print(f"✓ Saved: final_all_towers_metadata.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\nFinal embeddings: {all_embeddings.shape}")
print(f"\nTower breakdown:")
for tower, (start, end) in final_metadata['tower_ranges'].items():
    dims = end - start
    emb_slice = all_embeddings[:, start:end]
    print(f"  {tower:25s}: dims [{start:3d}:{end:3d}] = {dims:3d} dims "
          f"(mean={emb_slice.mean():.4f}, std={emb_slice.std():.4f})")

print("\n" + "="*80)
print("ALL TOWERS COMBINED - COMPLETE!")
print("="*80)
print(f"\nOutput file: final_all_towers_278d.parquet")
print(f"Ready for similarity analysis and validation!")
