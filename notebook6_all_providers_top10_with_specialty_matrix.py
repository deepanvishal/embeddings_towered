"""
NOTEBOOK 6: TOP 10 ALTERNATIVES FOR ALL PROVIDERS + SPECIALTY MATRIX
=====================================================================

Generates top 10 alternative providers for ALL 25,347 providers with:
- Full tower comparisons
- Specialty category distribution matrix
- Chunked processing to manage memory

Output:
1. all_providers_top10_alternatives.csv (253,470 rows with full details)
2. specialty_category_matrix.csv (distribution matrix)

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# ============================================================================
# LOAD ALL DATA FILES
# ============================================================================

# Labels
with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

print(f"Labels: {len(pin_to_label)}")

# Embeddings (278 dims - all towers)
embeddings_df = pd.read_parquet('final_all_towers_278d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

# Procedure data (ALL providers)
procedure_df = pd.read_parquet('procedure_df.parquet')
print(f"Procedure data: {procedure_df.shape}")

# Diagnosis data (ALL providers)
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
print(f"Diagnosis data: {diagnosis_df.shape}")

# Demographics (ALL providers)
demo_df = pd.read_parquet('demo_df.parquet')
print(f"Demographics: {demo_df.shape}")

# Place (ALL providers)
place_df = pd.read_parquet('place_df.parquet')
print(f"Place: {place_df.shape}")

# Cost (ALL providers)
cost_df = pd.read_parquet('cost_df.parquet')
print(f"Cost: {cost_df.shape}")

# PIN summary (ALL providers)
pin_df = pd.read_parquet('pin_df.parquet')
print(f"PIN summary: {pin_df.shape}")

# PIN names
pin_names_df = pd.read_parquet('all_pin_names.parquet')
all_pins_with_embeddings = embeddings_df['PIN'].values
pin_names_df = pin_names_df[pin_names_df['PIN'].isin(all_pins_with_embeddings)]
print(f"PIN names: {pin_names_df.shape}")

# Create PIN to name mapping
pin_to_name = dict(zip(pin_names_df['PIN'], pin_names_df['PIN_name']))

# Provider specialty
prov_spl_df = pd.read_parquet('prov_spl.parquet')
print(f"Provider specialty: {prov_spl_df.shape}")

# Create PIN to specialty mapping
pin_to_specialty = dict(zip(prov_spl_df['PIN'], prov_spl_df['srv_spclty_ctg_cd']))
print(f"PIN to specialty mapping: {len(pin_to_specialty)} providers")

# ============================================================================
# LOAD PROTOTYPE MODEL
# ============================================================================

print("\n" + "="*80)
print("LOADING PROTOTYPE MODEL")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
checkpoint = torch.load('trained_prototype_model.pth', map_location=device)

# Model architecture
class PrototypeWeightModel(torch.nn.Module):
    def __init__(self, n_prototypes, embedding_dim, n_towers=6):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.embedding_dim = embedding_dim
        self.n_towers = n_towers
        self.prototypes = torch.nn.Parameter(torch.randn(n_prototypes, embedding_dim) * 0.1)
        self.weight_profiles = torch.nn.Parameter(torch.ones(n_prototypes, n_towers))
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query_emb):
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        query_norm = F.normalize(query_emb, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarities = torch.matmul(query_norm, prototypes_norm.T)
        similarities = similarities / self.temperature
        prototype_weights = F.softmax(similarities, dim=1)
        tower_weights = torch.matmul(prototype_weights, self.weight_profiles)
        tower_weights = F.softmax(tower_weights, dim=1)
        
        if squeeze_output:
            tower_weights = tower_weights.squeeze(0)
        
        return tower_weights

# Initialize and load
model = PrototypeWeightModel(
    n_prototypes=checkpoint['n_prototypes'],
    embedding_dim=checkpoint['embedding_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

tower_dims = checkpoint['tower_dims']
print("✓ Model loaded")

# ============================================================================
# PREPARE EMBEDDINGS & MAPPINGS
# ============================================================================

print("\n" + "="*80)
print("PREPARING DATA STRUCTURES")
print("="*80)

# Get embedding columns
emb_cols = [col for col in embeddings_df.columns if col != 'PIN']

# Create mappings
pin_to_emb = {}
for _, row in embeddings_df.iterrows():
    pin = row['PIN']
    emb = row[emb_cols].values
    pin_to_emb[pin] = emb

embeddings_tensor = torch.FloatTensor(embeddings_df[emb_cols].values).to(device)
all_pins_list = embeddings_df['PIN'].values.tolist()
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins_list)}

# Procedure codes per PIN
procedure_summary = procedure_df.groupby('PIN').agg({
    'code': lambda x: set(x)
}).reset_index()
pin_to_procedure_codes = dict(zip(procedure_summary['PIN'], procedure_summary['code']))

# Diagnosis codes per PIN
diagnosis_summary = diagnosis_df.groupby('PIN').agg({
    'code': lambda x: set(x)
}).reset_index()
pin_to_diagnosis_codes = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['code']))

# Set PIN as index for linear towers
demo_df = demo_df.set_index('PIN')
place_df = place_df.set_index('PIN')
cost_df = cost_df.set_index('PIN')
pin_df = pin_df.set_index('PIN')

demo_cols = [col for col in demo_df.columns]
place_cols = [col for col in place_df.columns]
cost_cols = [col for col in cost_df.columns]
pin_cols = [col for col in pin_df.columns]

print(f"✓ Data structures ready")
print(f"  Total providers: {len(all_pins_list)}")
print(f"  Procedure data: {len(pin_to_procedure_codes)} providers")
print(f"  Diagnosis data: {len(pin_to_diagnosis_codes)} providers")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_similarity_manual(vec_a, vec_b):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)


def compute_tower_similarity(emb_a, emb_b, start_idx, end_idx):
    """Compute cosine similarity for a specific tower"""
    tower_a = emb_a[start_idx:end_idx]
    tower_b = emb_b[start_idx:end_idx]
    return cosine_similarity_manual(tower_a, tower_b)


def apply_tower_weights_vectorized(embeddings, tower_weights, tower_dims):
    """Apply tower weights to embeddings"""
    single = embeddings.dim() == 1
    if single:
        embeddings = embeddings.unsqueeze(0)
    
    if tower_weights.dim() == 1:
        tower_weights = tower_weights.unsqueeze(0)
    
    weighted = torch.zeros_like(embeddings)
    
    tower_list = ['procedures', 'diagnoses', 'demographics', 'place', 'cost', 'pin']
    for i, tower_name in enumerate(tower_list):
        start, end = tower_dims[tower_name]
        weighted[:, start:end] = embeddings[:, start:end] * tower_weights[:, i:i+1]
    
    if single:
        weighted = weighted.squeeze(0)
    
    return weighted


def find_top_k_similar(query_idx, embeddings_tensor, model, tower_dims, k=10, device='cpu'):
    """Find top K similar providers using prototype model"""
    
    query_emb = embeddings_tensor[query_idx].to(device)
    all_embs = embeddings_tensor.to(device)
    
    with torch.no_grad():
        # Predict weights
        weights = model(query_emb)
        
        # Apply weights to query
        weighted_query = apply_tower_weights_vectorized(query_emb, weights, tower_dims)
        
        # Apply weights to all candidates
        weighted_all = apply_tower_weights_vectorized(all_embs, weights, tower_dims)
        
        # Normalize
        weighted_query_norm = F.normalize(weighted_query.unsqueeze(0), p=2, dim=1)
        weighted_all_norm = F.normalize(weighted_all, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(weighted_query_norm, weighted_all_norm.T).squeeze()
        
        # Exclude self
        similarities[query_idx] = -1
        
        # Top K
        top_k_similarities, top_k_indices = torch.topk(similarities, k)
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy()

# ============================================================================
# GENERATE TOP 10 FOR ALL PROVIDERS (CHUNKED)
# ============================================================================

print("\n" + "="*80)
print("GENERATING TOP 10 ALTERNATIVES FOR ALL PROVIDERS")
print("="*80)

chunk_size = 1000
total_providers = len(all_pins_list)
num_chunks = (total_providers + chunk_size - 1) // chunk_size

print(f"Total providers: {total_providers:,}")
print(f"Chunk size: {chunk_size}")
print(f"Number of chunks: {num_chunks}")
print(f"Expected output rows: {total_providers * 10:,}")

# Process in chunks
all_results = []

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_providers)
    
    chunk_pins = all_pins_list[start_idx:end_idx]
    
    print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (providers {start_idx:,} to {end_idx:,})")
    
    chunk_results = []
    
    for query_pin in tqdm(chunk_pins, desc=f"Chunk {chunk_idx + 1}"):
        query_idx = pin_to_idx[query_pin]
        query_name = pin_to_name.get(query_pin, 'Unknown')
        query_label = pin_to_label.get(query_pin, 'Unlabeled')
        query_specialty = pin_to_specialty.get(query_pin, 'Unknown')
        
        # Find top 10
        top_10_indices, top_10_sims = find_top_k_similar(
            query_idx=query_idx,
            embeddings_tensor=embeddings_tensor,
            model=model,
            tower_dims=tower_dims,
            k=10,
            device=device
        )
        
        # Get query embedding
        query_emb = pin_to_emb[query_pin]
        
        # Process each recommendation
        for rank, (rec_idx, rec_sim) in enumerate(zip(top_10_indices, top_10_sims), 1):
            rec_pin = all_pins_list[rec_idx]
            rec_name = pin_to_name.get(rec_pin, 'Unknown')
            rec_label = pin_to_label.get(rec_pin, 'Unlabeled')
            rec_specialty = pin_to_specialty.get(rec_pin, 'Unknown')
            rec_emb = pin_to_emb[rec_pin]
            
            result = {
                'primary_pin': query_pin,
                'primary_pin_name': query_name,
                'primary_label': query_label,
                'primary_specialty': query_specialty,
                'rank': rank,
                'alternative_pin': rec_pin,
                'alternative_pin_name': rec_name,
                'alternative_label': rec_label,
                'alternative_specialty': rec_specialty,
                'prototype_weighted_similarity': rec_sim
            }
            
            # Procedure tower
            codes_a = pin_to_procedure_codes.get(query_pin, set())
            codes_b = pin_to_procedure_codes.get(rec_pin, set())
            
            result['primary_procedure_count'] = len(codes_a)
            result['alternative_procedure_count'] = len(codes_b)
            result['common_procedure_count'] = len(codes_a & codes_b)
            result['procedure_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['procedures'][0], tower_dims['procedures'][1]
            )
            
            # Diagnosis tower
            codes_a = pin_to_diagnosis_codes.get(query_pin, set())
            codes_b = pin_to_diagnosis_codes.get(rec_pin, set())
            
            result['primary_diagnosis_count'] = len(codes_a)
            result['alternative_diagnosis_count'] = len(codes_b)
            result['common_diagnosis_count'] = len(codes_a & codes_b)
            result['diagnosis_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['diagnoses'][0], tower_dims['diagnoses'][1]
            )
            
            # Demographics tower
            if query_pin in demo_df.index:
                for col in demo_cols:
                    result[f'primary_{col}'] = demo_df.loc[query_pin, col]
            else:
                for col in demo_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in demo_df.index:
                for col in demo_cols:
                    result[f'alternative_{col}'] = demo_df.loc[rec_pin, col]
            else:
                for col in demo_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['demographics_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['demographics'][0], tower_dims['demographics'][1]
            )
            
            # Place tower
            if query_pin in place_df.index:
                for col in place_cols:
                    result[f'primary_{col}'] = place_df.loc[query_pin, col]
            else:
                for col in place_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in place_df.index:
                for col in place_cols:
                    result[f'alternative_{col}'] = place_df.loc[rec_pin, col]
            else:
                for col in place_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['place_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['place'][0], tower_dims['place'][1]
            )
            
            # Cost tower
            if query_pin in cost_df.index:
                for col in cost_cols:
                    result[f'primary_{col}'] = cost_df.loc[query_pin, col]
            else:
                for col in cost_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in cost_df.index:
                for col in cost_cols:
                    result[f'alternative_{col}'] = cost_df.loc[rec_pin, col]
            else:
                for col in cost_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['cost_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['cost'][0], tower_dims['cost'][1]
            )
            
            # PIN tower
            if query_pin in pin_df.index:
                for col in pin_cols:
                    result[f'primary_{col}'] = pin_df.loc[query_pin, col]
            else:
                for col in pin_cols:
                    result[f'primary_{col}'] = np.nan
            
            if rec_pin in pin_df.index:
                for col in pin_cols:
                    result[f'alternative_{col}'] = pin_df.loc[rec_pin, col]
            else:
                for col in pin_cols:
                    result[f'alternative_{col}'] = np.nan
            
            result['pin_embedding_similarity'] = compute_tower_similarity(
                query_emb, rec_emb, tower_dims['pin'][0], tower_dims['pin'][1]
            )
            
            # Overall similarity
            result['overall_embedding_similarity'] = cosine_similarity_manual(query_emb, rec_emb)
            
            chunk_results.append(result)
    
    # Save chunk
    chunk_df = pd.DataFrame(chunk_results)
    
    if chunk_idx == 0:
        # First chunk - create file
        chunk_df.to_csv('all_providers_top10_alternatives.csv', index=False, mode='w')
    else:
        # Append to existing file
        chunk_df.to_csv('all_providers_top10_alternatives.csv', index=False, mode='a', header=False)
    
    print(f"✓ Chunk {chunk_idx + 1} saved ({len(chunk_results):,} rows)")
    
    # Collect all results for matrix calculation
    all_results.extend(chunk_results)
    
    # Cleanup
    del chunk_results, chunk_df
    gc.collect()

print("\n✓ All providers processed!")
print(f"✓ Total rows generated: {len(all_results):,}")

# ============================================================================
# CREATE SPECIALTY CATEGORY DISTRIBUTION MATRIX
# ============================================================================

print("\n" + "="*80)
print("CREATING SPECIALTY CATEGORY DISTRIBUTION MATRIX")
print("="*80)

# Convert to DataFrame for easier manipulation
results_df = pd.DataFrame(all_results)

# Filter out unknown specialties
matrix_df = results_df[
    (results_df['primary_specialty'] != 'Unknown') & 
    (results_df['alternative_specialty'] != 'Unknown')
].copy()

print(f"Pairs with known specialties: {len(matrix_df):,}")

# Create count matrix
count_matrix = pd.crosstab(
    matrix_df['primary_specialty'],
    matrix_df['alternative_specialty'],
    margins=True,
    margins_name='Total'
)

print(f"\nSpecialty count matrix shape: {count_matrix.shape}")

# Create percentage matrix (row-wise percentages)
# Each row sums to 100% - shows distribution of alternatives for each primary specialty
pct_matrix = count_matrix.div(count_matrix['Total'], axis=0) * 100
pct_matrix = pct_matrix.round(2)

# Save matrices
count_matrix.to_csv('specialty_category_count_matrix.csv')
print("✓ Saved: specialty_category_count_matrix.csv")

pct_matrix.to_csv('specialty_category_percentage_matrix.csv')
print("✓ Saved: specialty_category_percentage_matrix.csv")

# Create combined matrix for better readability
combined_matrix = count_matrix.copy()
for col in combined_matrix.columns:
    if col != 'Total':
        combined_matrix[col] = combined_matrix[col].astype(str) + ' (' + pct_matrix[col].astype(str) + '%)'

combined_matrix.to_csv('specialty_category_combined_matrix.csv')
print("✓ Saved: specialty_category_combined_matrix.csv (count + percentage)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal primary providers: {results_df['primary_pin'].nunique():,}")
print(f"Total alternative recommendations: {len(results_df):,}")

print(f"\nPrimary specialty distribution:")
primary_spec_dist = results_df.groupby('primary_specialty')['primary_pin'].nunique()
print(primary_spec_dist.sort_values(ascending=False))

print(f"\nAlternative specialty distribution:")
alt_spec_dist = results_df['alternative_specialty'].value_counts()
print(alt_spec_dist)

print(f"\nSame-specialty recommendations:")
same_specialty = (results_df['primary_specialty'] == results_df['alternative_specialty']).sum()
total_recs = len(results_df)
print(f"  Count: {same_specialty:,} ({same_specialty/total_recs:.1%})")

print(f"\nTop 10 specialty pairs (primary → alternative):")
top_pairs = results_df.groupby(['primary_specialty', 'alternative_specialty']).size().sort_values(ascending=False).head(10)
print(top_pairs)

print(f"\nSimilarity statistics:")
print(f"  Prototype-weighted similarity:")
print(f"    Mean: {results_df['prototype_weighted_similarity'].mean():.4f}")
print(f"    Std:  {results_df['prototype_weighted_similarity'].std():.4f}")
print(f"  Overall embedding similarity:")
print(f"    Mean: {results_df['overall_embedding_similarity'].mean():.4f}")
print(f"    Std:  {results_df['overall_embedding_similarity'].std():.4f}")

print("\n" + "="*80)
print("NOTEBOOK 6 COMPLETE")
print("="*80)
print(f"\n✓ Generated top 10 alternatives for {results_df['primary_pin'].nunique():,} providers")
print(f"✓ Total recommendations: {len(results_df):,}")
print(f"\nOutput files:")
print(f"  1. all_providers_top10_alternatives.csv")
print(f"  2. specialty_category_count_matrix.csv")
print(f"  3. specialty_category_percentage_matrix.csv")
print(f"  4. specialty_category_combined_matrix.csv")
