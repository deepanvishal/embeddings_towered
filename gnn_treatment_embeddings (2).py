import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import cudf

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("="*80)
print("GRAPH-BASED TREATMENT EMBEDDINGS - GPU OPTIMIZED")
print("="*80)

print(f"Loading data: {len(amt_smry_df):,} rows")

print("\nBuilding graph with GPU-accelerated groupby...")

# GPU-accelerated groupby with CuDF
df = cudf.DataFrame(amt_smry_df[['PIN', 'Medical_code', 'claims']])

grouped = df.groupby(['PIN', 'Medical_code'], as_index=False)['claims'].sum()
provider_totals = grouped.groupby('PIN')['claims'].transform('sum')
grouped['distribution'] = grouped['claims'] / provider_totals

code_provider_counts = grouped.groupby('Medical_code')['PIN'].nunique()
n_total_providers = grouped['PIN'].nunique()
idf_values = cudf.log(n_total_providers / code_provider_counts)
grouped['idf'] = grouped['Medical_code'].map(idf_values)
grouped['weight'] = grouped['distribution'] * (1.0 + grouped['idf'])
grouped['weight'] = grouped['weight'] / grouped['weight'].max()

grouped = grouped.to_pandas()

# Get unique providers and codes
all_providers = grouped['PIN'].unique()
all_codes = grouped['Medical_code'].unique()
provider_to_idx = {p: i for i, p in enumerate(all_providers)}
code_to_idx = {c: i for i, c in enumerate(all_codes)}
n_providers = len(all_providers)
n_codes = len(all_codes)

print(f"Providers: {n_providers:,}, Codes: {n_codes:,}, Edges: {len(grouped):,}")

# Map to indices
grouped['provider_idx'] = grouped['PIN'].map(provider_to_idx)
grouped['code_idx'] = grouped['Medical_code'].map(code_to_idx)

# Extract arrays
provider_indices = grouped['provider_idx'].values.astype(np.int64)
code_indices = grouped['code_idx'].values.astype(np.int64)
edge_weights = grouped['weight'].values.astype(np.float32)

# Build sparse adjacency matrices for vectorized aggregation
from scipy.sparse import coo_matrix

# Provider <- Code adjacency (for provider aggregation)
code_to_provider = coo_matrix(
    (edge_weights, (provider_indices, code_indices)),
    shape=(n_providers, n_codes),
    dtype=np.float32
)

# Code <- Provider adjacency (for code aggregation)
provider_to_code = coo_matrix(
    (edge_weights, (code_indices, provider_indices)),
    shape=(n_codes, n_providers),
    dtype=np.float32
)

# Convert to PyTorch sparse tensors
def scipy_to_torch_sparse(scipy_coo):
    indices = torch.from_numpy(np.vstack([scipy_coo.row, scipy_coo.col])).long()
    values = torch.from_numpy(scipy_coo.data)
    shape = scipy_coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

code_to_provider_sparse = scipy_to_torch_sparse(code_to_provider).to(device)
provider_to_code_sparse = scipy_to_torch_sparse(provider_to_code).to(device)

# Compute degree normalization once
provider_degrees = torch.sparse.sum(code_to_provider_sparse, dim=1).to_dense().unsqueeze(1) + 1e-8
code_degrees = torch.sparse.sum(provider_to_code_sparse, dim=1).to_dense().unsqueeze(1) + 1e-8

del grouped, provider_indices, code_indices, edge_weights, code_to_provider, provider_to_code
torch.cuda.empty_cache()

print("\nBuilding GNN model with vectorized aggregation...")

class TreatmentGNN(nn.Module):
    def __init__(self, n_providers, n_codes, embedding_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.provider_embedding = nn.Embedding(n_providers, embedding_dim)
        self.code_embedding = nn.Embedding(n_codes, embedding_dim)
        
        self.provider_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.code_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        nn.init.xavier_uniform_(self.provider_embedding.weight)
        nn.init.xavier_uniform_(self.code_embedding.weight)
    
    def forward(self, code_to_provider_sparse, provider_to_code_sparse, provider_degrees, code_degrees):
        provider_emb = self.provider_embedding.weight
        code_emb = self.code_embedding.weight
        
        # Vectorized aggregation using sparse matrix multiplication
        provider_messages = torch.sparse.mm(code_to_provider_sparse, code_emb)
        provider_messages = provider_messages / provider_degrees
        
        code_messages = torch.sparse.mm(provider_to_code_sparse, provider_emb)
        code_messages = code_messages / code_degrees
        
        # Transform
        provider_out = self.provider_mlp(torch.cat([provider_emb, provider_messages], dim=1))
        code_out = self.code_mlp(torch.cat([code_emb, code_messages], dim=1))
        
        return provider_out, code_out

model = TreatmentGNN(n_providers, n_codes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Rebuild edge tensors for link prediction
provider_code_edges = torch.from_numpy(
    np.vstack([code_to_provider_sparse.indices()[1].cpu().numpy(),
               code_to_provider_sparse.indices()[0].cpu().numpy()])
).to(device)
edge_weights_tensor = code_to_provider_sparse.values().to(device)

def link_prediction_loss(provider_emb, code_emb, pos_edges, edge_weights, batch_size=8192):
    num_edges = pos_edges.size(1)
    num_batches = (num_edges + batch_size - 1) // batch_size
    total_loss = 0.0
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_edges)
        
        batch_edges = pos_edges[:, start:end]
        batch_weights = edge_weights[start:end]
        
        # Vectorized scoring
        pos_src = provider_emb[batch_edges[0]]
        pos_tgt = code_emb[batch_edges[1]]
        pos_scores = (pos_src * pos_tgt).sum(dim=1) * batch_weights
        
        neg_tgt_idx = torch.randint(0, len(code_emb), (len(batch_edges[0]),), device=device)
        neg_src = pos_src
        neg_tgt = code_emb[neg_tgt_idx]
        neg_scores = (neg_src * neg_tgt).sum(dim=1)
        
        # Fused BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        
        total_loss += (pos_loss + neg_loss)
    
    return total_loss / num_batches

print("\nTraining GNN...")
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    if use_amp:
        with torch.cuda.amp.autocast():
            provider_emb, code_emb = model(code_to_provider_sparse, provider_to_code_sparse, 
                                          provider_degrees, code_degrees)
            loss = link_prediction_loss(provider_emb, code_emb, provider_code_edges, edge_weights_tensor)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        provider_emb, code_emb = model(code_to_provider_sparse, provider_to_code_sparse,
                                       provider_degrees, code_degrees)
        loss = link_prediction_loss(provider_emb, code_emb, provider_code_edges, edge_weights_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    scheduler.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

print("\nTraining completed.")

print("\nExtracting embeddings...")
model.eval()
with torch.no_grad():
    treatment_embeddings, _ = model(code_to_provider_sparse, provider_to_code_sparse,
                                   provider_degrees, code_degrees)
    treatment_embeddings = treatment_embeddings.cpu().numpy()

# Normalize
treatment_embeddings = treatment_embeddings / (np.linalg.norm(treatment_embeddings, axis=1, keepdims=True) + 1e-8)

# Create dataframe
embeddings_df = pd.DataFrame(
    treatment_embeddings,
    columns=[f'treatment_emb_{i}' for i in range(64)]
)
embeddings_df['PIN'] = all_providers

print(f"\nEmbeddings created: {embeddings_df.shape}")
print(embeddings_df.head())

# Save
embeddings_df.to_csv('treatment_embeddings_step1.csv', index=False)
torch.save({
    'model_state': model.state_dict(),
    'provider_to_idx': provider_to_idx,
    'code_to_idx': code_to_idx,
    'n_providers': n_providers,
    'n_codes': n_codes,
    'code_to_provider_sparse': code_to_provider_sparse.cpu(),
    'provider_to_code_sparse': provider_to_code_sparse.cpu(),
    'provider_degrees': provider_degrees.cpu(),
    'code_degrees': code_degrees.cpu()
}, 'treatment_gnn_step1.pt')

print("\nSaved: treatment_embeddings_step1.csv, treatment_gnn_step1.pt")
print("="*80)
