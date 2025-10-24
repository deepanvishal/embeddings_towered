"""
NOTEBOOK 5: INTERACTIVE TWO-PROVIDER COMPARISON
================================================

Interactive Jupyter widget to select two providers and view detailed
side-by-side comparison.

Features:
- Two dropdowns (Provider A and Provider B)
- Prototype-weighted similarity
- Tower weights
- Detailed procedure/diagnosis analysis
- Demographics, place, cost, PIN comparisons

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


def compute_prototype_similarity(pin_a, pin_b):
    """Compute prototype-weighted similarity (A as query)"""
    idx_a = pin_to_idx[pin_a]
    idx_b = pin_to_idx[pin_b]
    
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
    
    return similarity, weights.cpu().numpy()


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
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_two_providers(pin_a, pin_b):
    """Compare two providers in detail"""
    
    clear_output(wait=True)
    
    if not pin_a or not pin_b:
        print("Please select both Provider A and Provider B")
        return
    
    if pin_a == pin_b:
        print("Please select two different providers")
        return
    
    name_a = pin_to_name.get(pin_a, 'Unknown')
    name_b = pin_to_name.get(pin_b, 'Unknown')
    label_a = pin_to_label.get(pin_a, 'Unlabeled')
    label_b = pin_to_label.get(pin_b, 'Unlabeled')
    
    print("="*80)
    print("TWO-PROVIDER COMPARISON")
    print("="*80)
    
    print(f"\nProvider A: {name_a}")
    print(f"  PIN: {pin_a}")
    print(f"  Label: {label_a}")
    
    print(f"\nProvider B: {name_b}")
    print(f"  PIN: {pin_b}")
    print(f"  Label: {label_b}")
    
    print("\n" + "="*80)
    
    # Get embeddings
    emb_a = pin_to_emb[pin_a]
    emb_b = pin_to_emb[pin_b]
    
    # Compute prototype-weighted similarity
    print("\nComputing prototype-weighted similarity...")
    proto_sim, tower_weights = compute_prototype_similarity(pin_a, pin_b)
    
    print("\n" + "="*80)
    print("PROTOTYPE MODEL ANALYSIS")
    print("="*80)
    print(f"\nPrototype-Weighted Similarity (A as query): {proto_sim:.4f}")
    print("\nTower Weights (what model focused on when A is query):\n")
    tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN']
    for i, name in enumerate(tower_names):
        print(f"  {name:15s}: {tower_weights[i]:.4f}")
    
    # ========================================================================
    # PROCEDURE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PROCEDURE ANALYSIS")
    print("="*80)
    
    proc_sim = compute_tower_similarity(emb_a, emb_b, 
                                       tower_dims['procedures'][0], 
                                       tower_dims['procedures'][1])
    print(f"\nProcedure Embedding Similarity: {proc_sim:.4f}\n")
    
    proc_a = get_top_procedures(pin_a, 10)
    proc_b = get_top_procedures(pin_b, 10)
    common_proc = find_common_procedures(pin_a, pin_b, 10)
    
    print(f"Top 10 Procedures - {name_a}:")
    if len(proc_a) > 0:
        display(proc_a)
    else:
        print("No procedure data available\n")
    
    print(f"\nTop 10 Procedures - {name_b}:")
    if len(proc_b) > 0:
        display(proc_b)
    else:
        print("No procedure data available\n")
    
    print(f"\nTop 10 Common Procedures:")
    if len(common_proc) > 0:
        display(common_proc)
    else:
        print("No common procedures found\n")
    
    # ========================================================================
    # DIAGNOSIS ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("DIAGNOSIS ANALYSIS")
    print("="*80)
    
    diag_sim = compute_tower_similarity(emb_a, emb_b,
                                       tower_dims['diagnoses'][0],
                                       tower_dims['diagnoses'][1])
    print(f"\nDiagnosis Embedding Similarity: {diag_sim:.4f}\n")
    
    diag_a = get_top_diagnoses(pin_a, 10)
    diag_b = get_top_diagnoses(pin_b, 10)
    common_diag = find_common_diagnoses(pin_a, pin_b, 10)
    
    print(f"Top 10 Diagnoses - {name_a}:")
    if len(diag_a) > 0:
        display(diag_a)
    else:
        print("No diagnosis data available\n")
    
    print(f"\nTop 10 Diagnoses - {name_b}:")
    if len(diag_b) > 0:
        display(diag_b)
    else:
        print("No diagnosis data available\n")
    
    print(f"\nTop 10 Common Diagnoses:")
    if len(common_diag) > 0:
        display(common_diag)
    else:
        print("No common diagnoses found\n")
    
    # ========================================================================
    # DEMOGRAPHICS COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("DEMOGRAPHICS COMPARISON")
    print("="*80)
    
    demo_sim = compute_tower_similarity(emb_a, emb_b,
                                       tower_dims['demographics'][0],
                                       tower_dims['demographics'][1])
    print(f"\nDemographics Embedding Similarity: {demo_sim:.4f}\n")
    
    if pin_a in demo_df.index:
        demo_a = demo_df.loc[pin_a]
    else:
        demo_a = pd.Series(index=demo_cols, dtype=float)
    
    if pin_b in demo_df.index:
        demo_b = demo_df.loc[pin_b]
    else:
        demo_b = pd.Series(index=demo_cols, dtype=float)
    
    demo_comparison = create_comparison_table(demo_a, demo_b, name_a, name_b)
    if len(demo_comparison) > 0:
        display(demo_comparison)
    else:
        print("No demographic data available\n")
    
    # ========================================================================
    # PLACE OF SERVICE COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("PLACE OF SERVICE COMPARISON")
    print("="*80)
    
    place_sim = compute_tower_similarity(emb_a, emb_b,
                                        tower_dims['place'][0],
                                        tower_dims['place'][1])
    print(f"\nPlace Embedding Similarity: {place_sim:.4f}\n")
    
    if pin_a in place_df.index:
        place_a = place_df.loc[pin_a]
    else:
        place_a = pd.Series(index=place_cols, dtype=float)
    
    if pin_b in place_df.index:
        place_b = place_df.loc[pin_b]
    else:
        place_b = pd.Series(index=place_cols, dtype=float)
    
    place_comparison = create_comparison_table(place_a, place_b, name_a, name_b)
    if len(place_comparison) > 0:
        display(place_comparison)
    else:
        print("No place data available\n")
    
    # ========================================================================
    # COST CATEGORY COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COST CATEGORY COMPARISON")
    print("="*80)
    
    cost_sim = compute_tower_similarity(emb_a, emb_b,
                                       tower_dims['cost'][0],
                                       tower_dims['cost'][1])
    print(f"\nCost Embedding Similarity: {cost_sim:.4f}\n")
    
    if pin_a in cost_df.index:
        cost_a = cost_df.loc[pin_a]
    else:
        cost_a = pd.Series(index=cost_cols, dtype=float)
    
    if pin_b in cost_df.index:
        cost_b = cost_df.loc[pin_b]
    else:
        cost_b = pd.Series(index=cost_cols, dtype=float)
    
    cost_comparison = create_comparison_table(cost_a, cost_b, name_a, name_b)
    if len(cost_comparison) > 0:
        display(cost_comparison)
    else:
        print("No cost data available\n")
    
    # ========================================================================
    # PIN SUMMARY COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("PIN SUMMARY COMPARISON")
    print("="*80)
    
    pin_sim = compute_tower_similarity(emb_a, emb_b,
                                      tower_dims['pin'][0],
                                      tower_dims['pin'][1])
    print(f"\nPIN Embedding Similarity: {pin_sim:.4f}\n")
    
    if pin_a in pin_df.index:
        pin_a_data = pin_df.loc[pin_a]
    else:
        pin_a_data = pd.Series(index=pin_cols, dtype=float)
    
    if pin_b in pin_df.index:
        pin_b_data = pin_df.loc[pin_b]
    else:
        pin_b_data = pd.Series(index=pin_cols, dtype=float)
    
    pin_comparison = create_comparison_table(pin_a_data, pin_b_data, name_a, name_b)
    if len(pin_comparison) > 0:
        display(pin_comparison)
    else:
        print("No PIN summary data available\n")
    
    # ========================================================================
    # OVERALL SIMILARITY
    # ========================================================================
    overall_sim = cosine_similarity_manual(emb_a, emb_b)
    
    print("\n" + "="*80)
    print("OVERALL SIMILARITY")
    print("="*80)
    print(f"\nOverall Embedding Similarity (all 278 dims): {overall_sim:.4f}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    # Cleanup
    del proc_a, proc_b, common_proc
    del diag_a, diag_b, common_diag
    gc.collect()

# ============================================================================
# CREATE INTERACTIVE WIDGETS
# ============================================================================

print("\n" + "="*80)
print("CREATING INTERACTIVE WIDGETS")
print("="*80)

# Create dropdown options (PIN + Name)
dropdown_options = [('-- Select Provider --', None)] + \
                   [(f"{pin_to_name.get(pin, 'Unknown')} ({pin})", pin) 
                    for pin in all_pins_list]
dropdown_options = sorted(dropdown_options[1:], key=lambda x: x[0])
dropdown_options = [('-- Select Provider --', None)] + dropdown_options

# Create widgets
provider_a_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Provider A:',
    disabled=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': '150px'}
)

provider_b_dropdown = widgets.Dropdown(
    options=dropdown_options,
    description='Provider B:',
    disabled=False,
    layout=widgets.Layout(width='80%'),
    style={'description_width': '150px'}
)

compare_button = widgets.Button(
    description='Compare Providers',
    disabled=False,
    button_style='primary',
    layout=widgets.Layout(width='200px')
)

output_area = widgets.Output()

def on_compare_clicked(b):
    """Handle compare button click"""
    with output_area:
        pin_a = provider_a_dropdown.value
        pin_b = provider_b_dropdown.value
        compare_two_providers(pin_a, pin_b)

compare_button.on_click(on_compare_clicked)

# ============================================================================
# DISPLAY INTERFACE
# ============================================================================

print("\n" + "="*80)
print("INTERACTIVE TWO-PROVIDER COMPARISON SYSTEM")
print("="*80)
print(f"\nAvailable providers: {len(all_pins_list):,}")
print(f"Labeled providers: {len(pin_to_label):,}")
print(f"Unlabeled providers: {len(all_pins_list) - len(pin_to_label):,}")
print("\nSelect two providers and click 'Compare Providers'")
print("\nComparison includes:")
print("  - Prototype-weighted similarity")
print("  - Tower weights")
print("  - Procedure/diagnosis analysis with top 10 lists")
print("  - Demographics, place, cost, PIN comparisons")
print("\nSystem ready!")

display(widgets.VBox([
    provider_a_dropdown,
    provider_b_dropdown,
    compare_button,
    output_area
]))
