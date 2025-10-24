"""
NOTEBOOK 4: INTERACTIVE DEEP DIVE ANALYSIS - SINGLE PROVIDER
==============================================================

Interactive Jupyter widget to select a provider and view detailed analysis
of its top 10 recommendations from the prototype model.

Features:
- Dropdown with PIN + PIN name
- Shows tower weights from prototype model
- Detailed procedure/diagnosis analysis with top 10 lists
- Demographics, place, cost, PIN comparisons
- All 10 recommendations analyzed

Author: AI Assistant
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn.functional as F
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

# Code descriptions (procedures + diagnoses)
code_desc_df = pd.read_parquet('code_desc_df.parquet')
print(f"Code descriptions (before dedup): {code_desc_df.shape}")

# Handle duplicates - keep description with highest claims
code_desc_df = code_desc_df.sort_values('claims', ascending=False)
code_desc_df = code_desc_df.drop_duplicates(subset='code', keep='first')
print(f"Code descriptions (after dedup): {code_desc_df.shape}")

# Create code to description mapping
code_to_desc = dict(zip(code_desc_df['code'], code_desc_df['code_desc']))
print(f"Code to description mapping: {len(code_to_desc)} codes")

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
    'code': lambda x: list(x),
    'claims': lambda x: list(x)
}).reset_index()

pin_to_procedure_data = {}
for _, row in procedure_summary.iterrows():
    pin = row['PIN']
    codes = row['code']
    claims = row['claims']
    pin_to_procedure_data[pin] = pd.DataFrame({'code': codes, 'claims': claims})

# Diagnosis codes per PIN
diagnosis_summary = diagnosis_df.groupby('PIN').agg({
    'code': lambda x: list(x),
    'claims': lambda x: list(x)
}).reset_index()

pin_to_diagnosis_data = {}
for _, row in diagnosis_summary.iterrows():
    pin = row['PIN']
    codes = row['code']
    claims = row['claims']
    pin_to_diagnosis_data[pin] = pd.DataFrame({'code': codes, 'claims': claims})

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
print(f"  Total providers available: {len(all_pins_list)}")
print(f"  Procedure data: {len(pin_to_procedure_data)} providers")
print(f"  Diagnosis data: {len(pin_to_diagnosis_data)} providers")

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
    
    return top_k_indices.cpu().numpy(), top_k_similarities.cpu().numpy(), weights.cpu().numpy()


def get_top_procedures(pin, top_n=10):
    """Get top procedures for a provider with descriptions"""
    if pin not in pin_to_procedure_data:
        return pd.DataFrame(columns=['code', 'description', 'claims', '%'])
    
    df = pin_to_procedure_data[pin].copy()
    df = df.groupby('code', as_index=False)['claims'].sum()
    total_claims = df['claims'].sum()
    df['%'] = (df['claims'] / total_claims * 100).round(2)
    
    # Add descriptions
    df['description'] = df['code'].map(code_to_desc).fillna('Unknown')
    
    result = df.nlargest(top_n, 'claims')[['code', 'description', 'claims', '%']]
    return result


def get_top_diagnoses(pin, top_n=10):
    """Get top diagnoses for a provider with descriptions"""
    if pin not in pin_to_diagnosis_data:
        return pd.DataFrame(columns=['code', 'description', 'claims', '%'])
    
    df = pin_to_diagnosis_data[pin].copy()
    df = df.groupby('code', as_index=False)['claims'].sum()
    total_claims = df['claims'].sum()
    df['%'] = (df['claims'] / total_claims * 100).round(2)
    
    # Add descriptions
    df['description'] = df['code'].map(code_to_desc).fillna('Unknown')
    
    result = df.nlargest(top_n, 'claims')[['code', 'description', 'claims', '%']]
    return result


def find_common_procedures(pin_a, pin_b, top_n=10):
    """Find common procedures between two providers with descriptions"""
    proc_a = get_top_procedures(pin_a, 50)
    proc_b = get_top_procedures(pin_b, 50)
    
    if len(proc_a) == 0 or len(proc_b) == 0:
        return pd.DataFrame(columns=['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B'])
    
    common = proc_a.merge(proc_b, on=['code', 'description'], suffixes=('_A', '_B'), how='inner')
    common['total_claims'] = common['claims_A'] + common['claims_B']
    
    result = common.nlargest(top_n, 'total_claims')[['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B']]
    return result


def find_common_diagnoses(pin_a, pin_b, top_n=10):
    """Find common diagnoses between two providers with descriptions"""
    diag_a = get_top_diagnoses(pin_a, 50)
    diag_b = get_top_diagnoses(pin_b, 50)
    
    if len(diag_a) == 0 or len(diag_b) == 0:
        return pd.DataFrame(columns=['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B'])
    
    common = diag_a.merge(diag_b, on=['code', 'description'], suffixes=('_A', '_B'), how='inner')
    common['total_claims'] = common['claims_A'] + common['claims_B']
    
    result = common.nlargest(top_n, 'total_claims')[['code', 'description', 'claims_A', 'claims_B', '%_A', '%_B']]
    return result


def create_comparison_table(data_a, data_b, name_a, name_b):
    """Create side-by-side comparison table"""
    if len(data_a) == 0 and len(data_b) == 0:
        return pd.DataFrame()
    
    comparison = pd.DataFrame({
        'Metric': data_a.index if len(data_a) > 0 else data_b.index,
        name_a: data_a.values if len(data_a) > 0 else [np.nan] * len(data_b),
        name_b: data_b.values if len(data_b) > 0 else [np.nan] * len(data_a)
    })
    
    return comparison

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_provider(query_pin):
    """Perform deep dive analysis for a single provider"""
    
    clear_output(wait=True)
    
    query_name = pin_to_name.get(query_pin, 'Unknown')
    query_label = pin_to_label.get(query_pin, 'Unlabeled')
    
    print("="*80)
    print(f"DEEP DIVE ANALYSIS: {query_name}")
    print(f"PIN: {query_pin}")
    print(f"Label: {query_label}")
    print("="*80)
    
    # Get embeddings
    query_idx = pin_to_idx[query_pin]
    query_emb = pin_to_emb[query_pin]
    
    # Find top 10 recommendations
    print("\nFinding top 10 similar providers...")
    top_10_indices, top_10_sims, tower_weights = find_top_k_similar(
        query_idx=query_idx,
        embeddings_tensor=embeddings_tensor,
        model=model,
        tower_dims=tower_dims,
        k=10,
        device=device
    )
    
    print(f"✓ Found top 10 recommendations\n")
    
    # Show tower weights
    print("="*80)
    print("PROTOTYPE MODEL TOWER WEIGHTS")
    print("="*80)
    print("\nWhat the model focused on for this query:\n")
    tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN']
    for i, name in enumerate(tower_names):
        print(f"  {name:15s}: {tower_weights[i]:.4f}")
    
    # Analyze each recommendation
    for rank, (rec_idx, rec_sim) in enumerate(zip(top_10_indices, top_10_sims), 1):
        rec_pin = all_pins_list[rec_idx]
        rec_name = pin_to_name.get(rec_pin, 'Unknown')
        rec_label = pin_to_label.get(rec_pin, 'Unlabeled')
        rec_emb = pin_to_emb[rec_pin]
        
        print("\n" + "="*80)
        print(f"RECOMMENDATION #{rank}: {rec_name}")
        print(f"PIN: {rec_pin} | Label: {rec_label}")
        print(f"Prototype Weighted Similarity: {rec_sim:.4f}")
        print("="*80)
        
        # ====================================================================
        # PROCEDURE ANALYSIS
        # ====================================================================
        print(f"\n{'─'*80}")
        print("PROCEDURE ANALYSIS")
        print(f"{'─'*80}")
        
        proc_sim = compute_tower_similarity(query_emb, rec_emb, 
                                           tower_dims['procedures'][0], 
                                           tower_dims['procedures'][1])
        print(f"\nProcedure Embedding Similarity: {proc_sim:.4f}\n")
        
        proc_a = get_top_procedures(query_pin, 10)
        proc_b = get_top_procedures(rec_pin, 10)
        common_proc = find_common_procedures(query_pin, rec_pin, 10)
        
        print(f"Top 10 Procedures - {query_name}:")
        if len(proc_a) > 0:
            display(proc_a)
        else:
            print("No procedure data available\n")
        
        print(f"\nTop 10 Procedures - {rec_name}:")
        if len(proc_b) > 0:
            display(proc_b)
        else:
            print("No procedure data available\n")
        
        print(f"\nTop 10 Common Procedures:")
        if len(common_proc) > 0:
            display(common_proc)
        else:
            print("No common procedures found\n")
        
        # ====================================================================
        # DIAGNOSIS ANALYSIS
        # ====================================================================
        print(f"\n{'─'*80}")
        print("DIAGNOSIS ANALYSIS")
        print(f"{'─'*80}")
        
        diag_sim = compute_tower_similarity(query_emb, rec_emb,
                                           tower_dims['diagnoses'][0],
                                           tower_dims['diagnoses'][1])
        print(f"\nDiagnosis Embedding Similarity: {diag_sim:.4f}\n")
        
        diag_a = get_top_diagnoses(query_pin, 10)
        diag_b = get_top_diagnoses(rec_pin, 10)
        common_diag = find_common_diagnoses(query_pin, rec_pin, 10)
        
        print(f"Top 10 Diagnoses - {query_name}:")
        if len(diag_a) > 0:
            display(diag_a)
        else:
            print("No diagnosis data available\n")
        
        print(f"\nTop 10 Diagnoses - {rec_name}:")
        if len(diag_b) > 0:
            display(diag_b)
        else:
            print("No diagnosis data available\n")
        
        print(f"\nTop 10 Common Diagnoses:")
        if len(common_diag) > 0:
            display(common_diag)
        else:
            print("No common diagnoses found\n")
        
        # ====================================================================
        # DEMOGRAPHICS COMPARISON
        # ====================================================================
        print(f"\n{'─'*80}")
        print("DEMOGRAPHICS COMPARISON")
        print(f"{'─'*80}")
        
        demo_sim = compute_tower_similarity(query_emb, rec_emb,
                                           tower_dims['demographics'][0],
                                           tower_dims['demographics'][1])
        print(f"\nDemographics Embedding Similarity: {demo_sim:.4f}\n")
        
        if query_pin in demo_df.index:
            demo_a = demo_df.loc[query_pin]
        else:
            demo_a = pd.Series(index=demo_cols, dtype=float)
        
        if rec_pin in demo_df.index:
            demo_b = demo_df.loc[rec_pin]
        else:
            demo_b = pd.Series(index=demo_cols, dtype=float)
        
        demo_comparison = create_comparison_table(demo_a, demo_b, query_name, rec_name)
        if len(demo_comparison) > 0:
            display(demo_comparison)
        else:
            print("No demographic data available\n")
        
        # ====================================================================
        # PLACE OF SERVICE COMPARISON
        # ====================================================================
        print(f"\n{'─'*80}")
        print("PLACE OF SERVICE COMPARISON")
        print(f"{'─'*80}")
        
        place_sim = compute_tower_similarity(query_emb, rec_emb,
                                            tower_dims['place'][0],
                                            tower_dims['place'][1])
        print(f"\nPlace Embedding Similarity: {place_sim:.4f}\n")
        
        if query_pin in place_df.index:
            place_a = place_df.loc[query_pin]
        else:
            place_a = pd.Series(index=place_cols, dtype=float)
        
        if rec_pin in place_df.index:
            place_b = place_df.loc[rec_pin]
        else:
            place_b = pd.Series(index=place_cols, dtype=float)
        
        place_comparison = create_comparison_table(place_a, place_b, query_name, rec_name)
        if len(place_comparison) > 0:
            display(place_comparison)
        else:
            print("No place data available\n")
        
        # ====================================================================
        # COST CATEGORY COMPARISON
        # ====================================================================
        print(f"\n{'─'*80}")
        print("COST CATEGORY COMPARISON")
        print(f"{'─'*80}")
        
        cost_sim = compute_tower_similarity(query_emb, rec_emb,
                                           tower_dims['cost'][0],
                                           tower_dims['cost'][1])
        print(f"\nCost Embedding Similarity: {cost_sim:.4f}\n")
        
        if query_pin in cost_df.index:
            cost_a = cost_df.loc[query_pin]
        else:
            cost_a = pd.Series(index=cost_cols, dtype=float)
        
        if rec_pin in cost_df.index:
            cost_b = cost_df.loc[rec_pin]
        else:
            cost_b = pd.Series(index=cost_cols, dtype=float)
        
        cost_comparison = create_comparison_table(cost_a, cost_b, query_name, rec_name)
        if len(cost_comparison) > 0:
            display(cost_comparison)
        else:
            print("No cost data available\n")
        
        # ====================================================================
        # PIN SUMMARY COMPARISON
        # ====================================================================
        print(f"\n{'─'*80}")
        print("PIN SUMMARY COMPARISON")
        print(f"{'─'*80}")
        
        pin_sim = compute_tower_similarity(query_emb, rec_emb,
                                          tower_dims['pin'][0],
                                          tower_dims['pin'][1])
        print(f"\nPIN Embedding Similarity: {pin_sim:.4f}\n")
        
        if query_pin in pin_df.index:
            pin_a = pin_df.loc[query_pin]
        else:
            pin_a = pd.Series(index=pin_cols, dtype=float)
        
        if rec_pin in pin_df.index:
            pin_b = pin_df.loc[rec_pin]
        else:
            pin_b = pd.Series(index=pin_cols, dtype=float)
        
        pin_comparison = create_comparison_table(pin_a, pin_b, query_name, rec_name)
        if len(pin_comparison) > 0:
            display(pin_comparison)
        else:
            print("No PIN summary data available\n")
        
        # Overall similarity
        overall_sim = cosine_similarity_manual(query_emb, rec_emb)
        print(f"\n{'─'*80}")
        print(f"Overall Embedding Similarity (all 278 dims): {overall_sim:.4f}")
        print(f"{'─'*80}\n")
        
        # Cleanup
        del proc_a, proc_b, common_proc
        del diag_a, diag_b, common_diag
        gc.collect()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

# ============================================================================
# CREATE INTERACTIVE WIDGET
# ============================================================================

print("\n" + "="*80)
print("CREATING INTERACTIVE WIDGET")
print("="*80)

# Create dropdown options (PIN + Name)
dropdown_options = [(f"{pin_to_name.get(pin, 'Unknown')} ({pin})", pin) 
                    for pin in all_pins_list]
dropdown_options = sorted(dropdown_options, key=lambda x: x[0])

# Create widget
provider_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Select Provider:',
    disabled=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': '150px'}
)

output_area = widgets.Output()

def on_provider_selected(change):
    """Handle provider selection"""
    with output_area:
        selected_pin = change['new']
        if selected_pin:
            analyze_provider(selected_pin)

provider_dropdown.observe(on_provider_selected, names='value')

# ============================================================================
# DISPLAY INTERFACE
# ============================================================================

print("\n" + "="*80)
print("INTERACTIVE PROVIDER ANALYSIS SYSTEM")
print("="*80)
print(f"\nAvailable providers: {len(all_pins_list):,}")
print(f"Labeled providers: {len(pin_to_label):,}")
print(f"Unlabeled providers: {len(all_pins_list) - len(pin_to_label):,}")
print("\nSelect a provider from the dropdown to begin analysis.")
print("Analysis includes:")
print("  - Prototype model tower weights")
print("  - Top 10 recommendations")
print("  - Detailed procedure/diagnosis analysis")
print("  - Demographics, place, cost, PIN comparisons")
print("\nSystem ready!")

display(widgets.VBox([provider_dropdown, output_area]))
