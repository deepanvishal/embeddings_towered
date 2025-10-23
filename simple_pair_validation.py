import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random

print("Loading data...")

# Load procedure data
procedure_df = pd.read_parquet('procedure_df.parquet')

# Load labels
with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

# Load embeddings
embeddings_df = pd.read_parquet('hospital_embeddings_128d.parquet')

print(f"Procedure_df: {procedure_df.shape}")
print(f"Labeled PINs: {len(pin_to_label)}")
print(f"Embeddings: {embeddings_df.shape}")

# ============================================================================
# FILTER TO LABELED PINS ONLY
# ============================================================================

labeled_pins = list(pin_to_label.keys())
procedure_df_labeled = procedure_df[procedure_df['PIN'].isin(labeled_pins)]

print(f"\nFiltered procedure_df to labeled PINs: {procedure_df_labeled.shape}")

# ============================================================================
# CREATE 10,000 RANDOM PAIRS
# ============================================================================

print("\nCreating 10,000 random pairs...")

n_pairs = 10000
random_pairs = []

for _ in range(n_pairs):
    pin_a, pin_b = random.sample(labeled_pins, 2)
    random_pairs.append({'PINA': pin_a, 'PINB': pin_b})

random_pairs_df = pd.DataFrame(random_pairs)
print(f"Random pairs: {random_pairs_df.shape}")

# ============================================================================
# CALCULATE EMBEDDING SIMILARITY
# ============================================================================

print("\nCalculating embedding similarities...")

# Get embedding columns
emb_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]

# Create PIN to embedding mapping
pin_to_emb = {}
for _, row in embeddings_df.iterrows():
    pin = row['PIN']
    emb = row[emb_cols].values
    pin_to_emb[pin] = emb

# Calculate cosine similarity for each pair
similarities = []

for _, row in random_pairs_df.iterrows():
    pin_a = row['PINA']
    pin_b = row['PINB']
    
    emb_a = pin_to_emb[pin_a]
    emb_b = pin_to_emb[pin_b]
    
    # Cosine similarity
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    similarities.append(sim)

random_pairs_df['embedding_similarity'] = similarities

print(f"Calculated {len(similarities)} similarities")

# ============================================================================
# CALCULATE PROCEDURE STATISTICS
# ============================================================================

print("\nCalculating procedure statistics...")

# Group procedures by PIN
procedure_summary = procedure_df_labeled.groupby('PIN').agg({
    'code': lambda x: set(x),  # Unique codes
}).reset_index()
procedure_summary.columns = ['PIN', 'codes']

# Create PIN to codes mapping
pin_to_codes = dict(zip(procedure_summary['PIN'], procedure_summary['codes']))

# Calculate statistics for each pair
pin_a_total = []
pin_b_total = []
common_procedures = []

for _, row in random_pairs_df.iterrows():
    pin_a = row['PINA']
    pin_b = row['PINB']
    
    codes_a = pin_to_codes.get(pin_a, set())
    codes_b = pin_to_codes.get(pin_b, set())
    
    pin_a_total.append(len(codes_a))
    pin_b_total.append(len(codes_b))
    common_procedures.append(len(codes_a & codes_b))

random_pairs_df['total_pina_procedures'] = pin_a_total
random_pairs_df['total_pinb_procedures'] = pin_b_total
random_pairs_df['common_procedures'] = common_procedures

# ============================================================================
# ADD LABELS
# ============================================================================

print("\nAdding labels...")

random_pairs_df['pina_label'] = random_pairs_df['PINA'].map(pin_to_label)
random_pairs_df['pinb_label'] = random_pairs_df['PINB'].map(pin_to_label)

# ============================================================================
# FINAL OUTPUT
# ============================================================================

# Reorder columns
final_df = random_pairs_df[[
    'PINA', 
    'pina_label', 
    'PINB', 
    'pinb_label', 
    'total_pina_procedures', 
    'total_pinb_procedures', 
    'common_procedures', 
    'embedding_similarity'
]]

# Save
final_df.to_csv('random_pairs_validation.csv', index=False)

print(f"\n✓ Saved: random_pairs_validation.csv")
print(f"  Shape: {final_df.shape}")
print(f"\nFirst 5 rows:")
print(final_df.head())

print("\n" + "="*80)
print("DONE")
print("="*80)
