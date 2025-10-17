import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("GRAPHSAGE AUTOENCODER - MEMORY EFFICIENT")
print("="*80)
print(f"Device: {device}\n")

print("Step 1: Processing data...")

amt_smry_df['dollar_value'] = pd.to_numeric(amt_smry_df['dollar_value'], errors='coerce').astype(np.float32)
amt_smry_df = amt_smry_df.dropna(subset=['dollar_value'])

grouped = amt_smry_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

feature_dicts = []
pin_list = []

for pin in grouped['PIN'].unique():
    pin_data = grouped[grouped['PIN'] == pin]
    spending_dict = dict(zip(pin_data['Medical_code'], pin_data['dollar_value']))
    feature_dicts.append(spending_dict)
    pin_list.append(pin)

vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
amt_smry_sparse = vectorizer.fit_transform(feature_dicts)

row_sums = np.array(amt_smry_sparse.sum(axis=1)).flatten()

if (row_sums == 0).any():
    non_zero_mask = row_sums > 0
    amt_smry_sparse = amt_smry_sparse[non_zero_mask]
    pin_list = [pin_list[i] for i in range(len(pin_list)) if non_zero_mask[i]]
    row_sums = row_sums[non_zero_mask]

row_sums_inv = 1.0 / row_sums
row_sums_inv = row_sums_inv.reshape(-1, 1)
amt_smry_normalized = amt_smry_sparse.multiply(row_sums_inv)
amt_smry_vectors = amt_smry_normalized.toarray().astype(np.float32)

n_providers = len(pin_list)
n_codes = amt_smry_vectors.shape[1]

print(f"Providers: {n_providers:,}, Codes: {n_codes:,}")

print("\nStep 2: Computing TF-IDF weights...")

code_doc_freq = (amt_smry_vectors > 0).sum(axis=0)
idf = np.log(n_providers / (code_doc_freq + 1))

amt_smry_tfidf = amt_smry_vectors * idf
max_weight = amt_smry_tfidf.max()
amt_smry_tfidf = amt_smry_tfidf / max_weight

print(f"IDF computed for {n_codes} codes")

print("\nStep 3: Building bipartite graph...")

edge_list = []
edge_weights = []

for i, provider_vector in enumerate(amt_smry_tfidf):
    nonzero_indices = np.nonzero(provider_vector)[0]
    for code_idx in nonzero_indices:
        edge_list.append([i, n_providers + code_idx])
        edge_weights.append(provider_vector[code_idx])

edge_index = torch.LongTensor(edge_list).t()
# Make bidirectional
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

edge_weight = torch.FloatTensor(edge_weights)
edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

# NO HUGE IDENTITY MATRIX! Just store the graph structure
data = Data(
    edge_index=edge_index,
    edge_attr=edge_weight,
    num_nodes=n_providers + n_codes
)

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

del amt_smry_sparse, amt_smry_normalized, amt_smry_vectors
torch.cuda.empty_cache()

print("\nStep 4: Creating mini-batch loader...")

# Mini-batch loader - only loads small subgraphs at a time!
train_loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],  # Sample 15 neighbors in layer 1, 10 in layer 2
    batch_size=1024,         # Process 1024 nodes at a time
    shuffle=True,
    num_workers=0
)

print(f"Batch size: 1024 nodes per batch")

print("\nStep 5: Building GraphSAGE Model...")

class GraphSAGE(nn.Module):
    def __init__(self, num_nodes, hidden_dim=256, embedding_dim=128):
        super().__init__()
        # Embedding layer instead of huge identity matrix!
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        
    def forward(self, node_ids, edge_index, edge_weight=None):
        x = self.embedding(node_ids)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

model = GraphSAGE(
    num_nodes=n_providers + n_codes,
    hidden_dim=256,
    embedding_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 6: Training GraphSAGE (10 epochs with mini-batches)...")

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get node IDs for this batch
        node_ids = torch.arange(batch.num_nodes, device=device)
        
        # Forward pass
        z = model(node_ids, batch.edge_index, batch.edge_attr)
        
        # Decode edges and compute loss
        reconstructed = model.decode(z, batch.edge_index)
        loss = F.mse_loss(reconstructed, batch.edge_attr)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

print("\nStep 7: Extracting provider embeddings...")

# Extract embeddings in batches to avoid OOM
model.eval()
all_embeddings = []

with torch.no_grad():
    # Process in chunks to be safe
    chunk_size = 5000
    for start_idx in range(0, n_providers + n_codes, chunk_size):
        end_idx = min(start_idx + chunk_size, n_providers + n_codes)
        node_ids = torch.arange(start_idx, end_idx, device=device)
        
        # Use only the edges relevant to these nodes
        mask = (data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx)
        mini_edge_index = data.edge_index[:, mask].to(device) - start_idx
        mini_edge_attr = data.edge_attr[mask].to(device)
        
        # Get embeddings for this chunk
        chunk_embeddings = model(node_ids, mini_edge_index, mini_edge_attr)
        all_embeddings.append(chunk_embeddings.cpu().numpy())

# Combine all chunks
all_embeddings = np.vstack(all_embeddings)
provider_embeddings = all_embeddings[:n_providers]

# Normalize
provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = pin_list

print(f"\nEmbeddings created: {embeddings_df.shape}")
print(embeddings_df.head())

print("\nStep 8: Quality check...")

emb_cols = [f'emb_{i}' for i in range(128)]
emb_std = embeddings_df[emb_cols].std().mean()
print(f"Embedding std: {emb_std:.6f} (good if > 0.05)")

sims = []
for _ in range(100):
    pin1, pin2 = np.random.choice(embeddings_df['PIN'].values, 2, replace=False)
    emb1 = embeddings_df[embeddings_df['PIN'] == pin1][emb_cols].values
    emb2 = embeddings_df[embeddings_df['PIN'] == pin2][emb_cols].values
    sims.append(np.dot(emb1, emb2.T)[0][0])

print(f"Random pair similarity: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")
print(f"(Good: mean ~0.0-0.3, std > 0.1)")

print("\nStep 9: Saving...")

embeddings_df.to_csv('graphsage_embeddings.csv', index=False)

torch.save({
    'model_state': model.state_dict(),
    'pin_list': pin_list,
    'n_providers': n_providers,
    'n_codes': n_codes,
    'vectorizer': vectorizer
}, 'graphsage_model.pt')

print("Saved: graphsage_embeddings.csv, graphsage_model.pt")
print("="*80)
print("\nMEMORY SAVINGS:")
print("- No huge identity matrix (saved ~900MB for 15k nodes)")
print("- Mini-batch training (only processes 1024 nodes at a time)")
print("- Embedding layer instead of one-hot encoding")
print("="*80)
