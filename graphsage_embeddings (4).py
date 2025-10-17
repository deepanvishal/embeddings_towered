import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
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
print("GRAPHSAGE AUTOENCODER - MEMORY EFFICIENT (No Extra Dependencies)")
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

# Store on CPU initially - we'll move batches to GPU as needed
data = Data(
    edge_index=edge_index,
    edge_attr=edge_weight,
    num_nodes=n_providers + n_codes
)

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

del amt_smry_sparse, amt_smry_normalized, amt_smry_vectors
torch.cuda.empty_cache()

print("\nStep 4: Building GraphSAGE Model with manual batching...")

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

model = GraphSAGE(
    num_nodes=n_providers + n_codes,
    hidden_dim=256,
    embedding_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 5: Training with edge batching (10 epochs)...")

# Create edge batches instead of node batches
batch_size = 10000  # edges per batch
num_edges = data.num_edges
num_batches = (num_edges + batch_size - 1) // batch_size

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Shuffle edges
    perm = torch.randperm(num_edges)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_edges)
        batch_edges = perm[start_idx:end_idx]
        
        # Get batch edges and weights
        batch_edge_index = data.edge_index[:, batch_edges].to(device)
        batch_edge_attr = data.edge_attr[batch_edges].to(device)
        
        # Get unique nodes in this batch
        unique_nodes = torch.unique(batch_edge_index)
        
        # Create node mapping (global ID -> local batch ID)
        node_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_map[unique_nodes] = torch.arange(len(unique_nodes))
        
        # Remap edges to local indices
        remapped_edges = torch.stack([
            node_map[batch_edge_index[0]],
            node_map[batch_edge_index[1]]
        ]).to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings for the unique nodes (using global IDs)
        x = model.embedding(unique_nodes.to(device))
        
        # Apply graph convolutions (using local remapped edges)
        x = model.conv1(x, remapped_edges, batch_edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=model.training)
        z = model.conv2(x, remapped_edges, batch_edge_attr)
        
        # Decode and compute loss
        src, dst = remapped_edges
        reconstructed = (z[src] * z[dst]).sum(dim=1)
        loss = F.mse_loss(reconstructed, batch_edge_attr)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f} ({num_batches} batches)")

print("\nStep 6: Extracting provider embeddings (in chunks)...")

model.eval()
provider_embeddings_list = []

chunk_size = 2000
with torch.no_grad():
    for start_idx in range(0, n_providers, chunk_size):
        end_idx = min(start_idx + chunk_size, n_providers)
        
        # Get edges connected to these provider nodes
        mask = ((data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx)) | \
               ((data.edge_index[1] >= start_idx) & (data.edge_index[1] < end_idx))
        
        chunk_edge_index = data.edge_index[:, mask].to(device)
        chunk_edge_attr = data.edge_attr[mask].to(device)
        
        # Get all unique nodes involved in these edges
        all_nodes = torch.unique(chunk_edge_index)
        
        # Create mapping (global ID -> local batch ID)
        node_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_map[all_nodes] = torch.arange(len(all_nodes))
        
        # Remap edges to local indices
        remapped = torch.stack([
            node_map[chunk_edge_index[0]],
            node_map[chunk_edge_index[1]]
        ]).to(device)
        
        # Get embeddings for all involved nodes (using global IDs)
        x = model.embedding(all_nodes.to(device))
        
        # Apply graph convolutions (using local remapped edges)
        x = model.conv1(x, remapped, chunk_edge_attr)
        x = F.relu(x)
        z = model.conv2(x, remapped, chunk_edge_attr)
        
        # Extract only the provider nodes we care about (from start_idx to end_idx)
        provider_mask = (all_nodes >= start_idx) & (all_nodes < end_idx)
        chunk_embeddings = z[provider_mask].cpu().numpy()
        provider_embeddings_list.append(chunk_embeddings)
        
        if (start_idx // chunk_size) % 5 == 0:
            print(f"  Processed {end_idx}/{n_providers} providers...")

provider_embeddings = np.vstack(provider_embeddings_list)

# Normalize
provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = pin_list

print(f"\nEmbeddings created: {embeddings_df.shape}")
print(embeddings_df.head())

print("\nStep 7: Quality check...")

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

print("\nStep 8: Saving...")

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
print("- No huge identity matrix (uses Embedding layer)")
print("- Edge-batch training (processes 10k edges at a time)")
print("- Chunked inference (processes 2k nodes at a time)")
print("- No pyg-lib or torch-sparse needed!")
print("="*80)
