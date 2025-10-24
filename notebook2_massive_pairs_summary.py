"""
NOTEBOOK 2: MASSIVE PAIRS SUMMARY - ALL TOWER COMPARISONS
===========================================================

Creates comprehensive comparison table for 10,000 random provider pairs:
- Procedure counts & similarity
- Diagnosis counts & similarity  
- Demographics values & similarity
- Place values & similarity
- Cost values & similarity
- PIN values & similarity
- Overall embedding similarity
- Prototype-weighted similarity

Output: One CSV with ~60-80 columns

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# ============================================================================
# LOAD ALL DATA FILES
# ============================================================================

# Labels
with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

labeled_pins = list(pin_to_label.keys())
print(f"Labeled providers: {len(labeled_pins)}")

# Embeddings (278 dims - all towers)
embeddings_df = pd.read_parquet('final_all_towers_278d.parquet')
print(f"Embeddings: {embeddings_df.shape}")

# Procedure data
procedure_df = pd.read_parquet('procedure_df.parquet')
procedure_df_labeled = procedure_df[procedure_df['PIN'].isin(labeled_pins)]
print(f"Procedure data: {procedure_df_labeled.shape}")

# Diagnosis data
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
diagnosis_df_labeled = diagnosis_df[diagnosis_df['PIN'].isin(labeled_pins)]
print(f"Diagnosis data: {diagnosis_df_labeled.shape}")

# Demographics
demo_df = pd.read_parquet('demo_df.parquet')
demo_df_labeled = demo_df[demo_df['PIN'].isin(labeled_pins)]
print(f"Demographics: {demo_df_labeled.shape}")

# Place
place_df = pd.read_parquet('place_df.parquet')
place_df_labeled = place_df[place_df['PIN'].isin(labeled_pins)]
print(f"Place: {place_df_labeled.shape}")

# Cost
cost_df = pd.read_parquet('cost_df.parquet')
cost_df_labeled = cost_df[cost_df['PIN'].isin(labeled_pins)]
print(f"Cost: {cost_df_labeled.shape}")

# PIN summary
pin_df = pd.read_parquet('pin_df.parquet')
pin_df_labeled = pin_df[pin_df['PIN'].isin(labeled_pins)]
print(f"PIN summary: {pin_df_labeled.shape}")

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

# Model architecture (copy from training)
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
# PREPARE EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("PREPARING EMBEDDINGS")
print("="*80)

# Get embedding columns
emb_cols = [col for col in embeddings_df.columns if col != 'PIN']
print(f"Embedding columns: {len(emb_cols)}")

# Create mappings
pin_to_emb = {}
for _, row in embeddings_df.iterrows():
    pin = row['PIN']
    emb = row[emb_cols].values
    pin_to_emb[pin] = emb

embeddings_tensor = torch.FloatTensor(embeddings_df[emb_cols].values).to(device)
pin_to_idx = {pin: idx for idx, pin in enumerate(embeddings_df['PIN'].values)}

print(f"✓ Created PIN to embedding mapping: {len(pin_to_emb)} providers")

# ============================================================================
# PREPARE PROCEDURE/DIAGNOSIS CODE SETS
# ============================================================================

print("\n" + "="*80)
print("PREPARING CODE SETS")
print("="*80)

# Procedure codes per PIN
procedure_summary = procedure_df_labeled.groupby('PIN').agg({
    'code': lambda x: set(x)
}).reset_index()
pin_to_procedure_codes = dict(zip(procedure_summary['PIN'], procedure_summary['code']))
print(f"✓ Procedure codes: {len(pin_to_procedure_codes)} providers")

# Diagnosis codes per PIN
diagnosis_summary = diagnosis_df_labeled.groupby('PIN').agg({
    'code': lambda x: set(x)
}).reset_index()
pin_to_diagnosis_codes = dict(zip(diagnosis_summary['PIN'], diagnosis_summary['code']))
print(f"✓ Diagnosis codes: {len(pin_to_diagnosis_codes)} providers")

# ============================================================================
# PREPARE LINEAR TOWER DATAFRAMES
# ============================================================================

print("\n" + "="*80)
print("PREPARING LINEAR TOWER DATA")
print("="*80)

# Set PIN as index for easy lookup
demo_df_labeled = demo_df_labeled.set_index('PIN')
place_df_labeled = place_df_labeled.set_index('PIN')
cost_df_labeled = cost_df_labeled.set_index('PIN')
pin_df_labeled = pin_df_labeled.set_index('PIN')

# Get column names (excluding PIN)
demo_cols = [col for col in demo_df_labeled.columns]
place_cols = [col for col in place_df_labeled.columns]
cost_cols = [col for col in cost_df_labeled.columns]
pin_cols = [col for col in pin_df_labeled.columns]

print(f"Demographics columns: {demo_cols}")
print(f"Place columns: {place_cols}")
print(f"Cost columns: {cost_cols}")
print(f"PIN columns: {pin_cols}")

# ============================================================================
# CREATE 10,000 RANDOM PAIRS
# ============================================================================

print("\n" + "="*80)
print("CREATING 10,000 RANDOM PAIRS")
print("="*80)

n_pairs = 10000
random_pairs = []

for _ in range(n_pairs):
    pin_a, pin_b = random.sample(labeled_pins, 2)
    random_pairs.append({'PINA': pin_a, 'PINB': pin_b})

random_pairs_df = pd.DataFrame(random_pairs)
print(f"✓ Created {len(random_pairs_df)} random pairs")

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


def compute_prototype_similarity(idx_a, idx_b, embeddings_tensor, model, tower_dims, device):
    """Compute prototype-weighted similarity (A as query)"""
    emb_a = embeddings_tensor[idx_a].to(device)
    emb_b = embeddings_tensor[idx_b].to(device)
    
    with torch.no_grad():
        # Predict weights based on A
        weights = model(emb_a)
        
        # Apply weights
        weighted_a = apply_tower_weights_vectorized(emb_a, weights, tower_dims)
        weighted_b = apply_tower_weights_vectorized(emb_b, weights, tower_dims)
        
        # Normalize
        weighted_a_norm = F.normalize(weighted_a.unsqueeze(0), p=2, dim=1)
        weighted_b_norm = F.normalize(weighted_b.unsqueeze(0), p=2, dim=1)
        
        # Cosine similarity
        similarity = torch.matmul(weighted_a_norm, weighted_b_norm.T).squeeze().item()
    
    return similarity

# ============================================================================
# PROCESS ALL PAIRS
# ============================================================================

print("\n" + "="*80)
print("PROCESSING ALL PAIRS")
print("="*80)

results = []

for _, row in tqdm(random_pairs_df.iterrows(), total=len(random_pairs_df), desc="Processing pairs"):
    pin_a = row['PINA']
    pin_b = row['PINB']
    
    result = {
        'PINA': pin_a,
        'PINA_label': pin_to_label[pin_a],
        'PINB': pin_b,
        'PINB_label': pin_to_label[pin_b]
    }
    
    # Get embeddings
    emb_a = pin_to_emb[pin_a]
    emb_b = pin_to_emb[pin_b]
    
    # ========================================================================
    # PROCEDURE TOWER
    # ========================================================================
    codes_a = pin_to_procedure_codes.get(pin_a, set())
    codes_b = pin_to_procedure_codes.get(pin_b, set())
    
    result['provA_procedure_count'] = len(codes_a)
    result['provB_procedure_count'] = len(codes_b)
    result['common_procedure_count'] = len(codes_a & codes_b)
    result['procedure_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['procedures'][0], tower_dims['procedures'][1]
    )
    
    # ========================================================================
    # DIAGNOSIS TOWER
    # ========================================================================
    codes_a = pin_to_diagnosis_codes.get(pin_a, set())
    codes_b = pin_to_diagnosis_codes.get(pin_b, set())
    
    result['provA_diagnosis_count'] = len(codes_a)
    result['provB_diagnosis_count'] = len(codes_b)
    result['common_diagnosis_count'] = len(codes_a & codes_b)
    result['diagnosis_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['diagnoses'][0], tower_dims['diagnoses'][1]
    )
    
    # ========================================================================
    # DEMOGRAPHICS TOWER
    # ========================================================================
    if pin_a in demo_df_labeled.index:
        for col in demo_cols:
            result[f'provA_{col}'] = demo_df_labeled.loc[pin_a, col]
    else:
        for col in demo_cols:
            result[f'provA_{col}'] = np.nan
    
    if pin_b in demo_df_labeled.index:
        for col in demo_cols:
            result[f'provB_{col}'] = demo_df_labeled.loc[pin_b, col]
    else:
        for col in demo_cols:
            result[f'provB_{col}'] = np.nan
    
    result['demographics_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['demographics'][0], tower_dims['demographics'][1]
    )
    
    # ========================================================================
    # PLACE TOWER
    # ========================================================================
    if pin_a in place_df_labeled.index:
        for col in place_cols:
            result[f'provA_{col}'] = place_df_labeled.loc[pin_a, col]
    else:
        for col in place_cols:
            result[f'provA_{col}'] = np.nan
    
    if pin_b in place_df_labeled.index:
        for col in place_cols:
            result[f'provB_{col}'] = place_df_labeled.loc[pin_b, col]
    else:
        for col in place_cols:
            result[f'provB_{col}'] = np.nan
    
    result['place_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['place'][0], tower_dims['place'][1]
    )
    
    # ========================================================================
    # COST TOWER
    # ========================================================================
    if pin_a in cost_df_labeled.index:
        for col in cost_cols:
            result[f'provA_{col}'] = cost_df_labeled.loc[pin_a, col]
    else:
        for col in cost_cols:
            result[f'provA_{col}'] = np.nan
    
    if pin_b in cost_df_labeled.index:
        for col in cost_cols:
            result[f'provB_{col}'] = cost_df_labeled.loc[pin_b, col]
    else:
        for col in cost_cols:
            result[f'provB_{col}'] = np.nan
    
    result['cost_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['cost'][0], tower_dims['cost'][1]
    )
    
    # ========================================================================
    # PIN TOWER
    # ========================================================================
    if pin_a in pin_df_labeled.index:
        for col in pin_cols:
            result[f'provA_{col}'] = pin_df_labeled.loc[pin_a, col]
    else:
        for col in pin_cols:
            result[f'provA_{col}'] = np.nan
    
    if pin_b in pin_df_labeled.index:
        for col in pin_cols:
            result[f'provB_{col}'] = pin_df_labeled.loc[pin_b, col]
    else:
        for col in pin_cols:
            result[f'provB_{col}'] = np.nan
    
    result['pin_embedding_similarity'] = compute_tower_similarity(
        emb_a, emb_b, tower_dims['pin'][0], tower_dims['pin'][1]
    )
    
    # ========================================================================
    # OVERALL EMBEDDING SIMILARITY (all 278 dims)
    # ========================================================================
    result['overall_embedding_similarity'] = cosine_similarity_manual(emb_a, emb_b)
    
    # ========================================================================
    # PROTOTYPE WEIGHTED SIMILARITY (A as query)
    # ========================================================================
    idx_a = pin_to_idx[pin_a]
    idx_b = pin_to_idx[pin_b]
    result['prototype_weighted_similarity'] = compute_prototype_similarity(
        idx_a, idx_b, embeddings_tensor, model, tower_dims, device
    )
    
    results.append(result)

# ============================================================================
# CREATE FINAL DATAFRAME
# ============================================================================

print("\n" + "="*80)
print("CREATING FINAL DATAFRAME")
print("="*80)

final_df = pd.DataFrame(results)

print(f"✓ Final dataframe shape: {final_df.shape}")
print(f"  Total columns: {len(final_df.columns)}")

# ============================================================================
# SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

final_df.to_csv('massive_pairs_summary.csv', index=False)
print(f"✓ Saved: massive_pairs_summary.csv")

final_df.to_parquet('massive_pairs_summary.parquet', index=False)
print(f"✓ Saved: massive_pairs_summary.parquet")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nDataframe shape: {final_df.shape}")
print(f"\nFirst 3 rows:")
print(final_df.head(3))

print(f"\nColumn list:")
for i, col in enumerate(final_df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSimilarity statistics:")
sim_cols = [col for col in final_df.columns if 'similarity' in col]
for col in sim_cols:
    print(f"  {col:40s}: mean={final_df[col].mean():.4f}, std={final_df[col].std():.4f}")

print("\n" + "="*80)
print("NOTEBOOK 2 COMPLETE")
print("="*80)
print(f"\n✓ Processed {len(final_df)} pairs")
print(f"✓ Output: massive_pairs_summary.csv ({final_df.shape[1]} columns)")
