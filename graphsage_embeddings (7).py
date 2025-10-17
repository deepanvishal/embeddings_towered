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

set_seeds(42)

print("="*80)
print("GRAPHSAGE - CPU ONLY VERSION")
print("="*80)

# ============================================================================
# SAMPLE DATA GENERATION (Comment this out when using real data)
# ============================================================================
print("Generating sample data for testing...")
np.random.seed(42)

n_sample_providers = 500
n_sample_codes = 200
n_transactions = 5000

sample_pins = [f"PIN_{i:04d}" for i in range(n_sample_providers)]
sample_codes = [f"CODE_{i:03d}" for i in range(n_sample_codes)]

pins = np.random.choice(sample_pins, n_transactions)
codes = np.random.choice(sample_codes, n_transactions)
values = np.random.lognormal(mean=5, sigma=2, size=n_transactions)

amt_smry_df = pd.DataFrame({
    'PIN': pins,
    'Medical_code': codes,
    'dollar_value': values
})

print(f"Sample data: {len(amt_smry_df)} transactions")
print(f"Unique PINs: {amt_smry_df['PIN'].nunique()}")
print(f"Unique Codes: {amt_smry_df['Medical_code'].nunique()}")
print()
# ============================================================================
# END SAMPLE DATA - Use your real amt_smry_df below this line
# ============================================================================

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
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

edge_weight = torch.FloatTensor(edge_weights)
edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

data = Data(
    edge_index=edge_index,
    edge_attr=edge_weight,
    num_nodes=n_providers + n_codes
)

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

del amt_smry_sparse, amt_smry_normalized, amt_smry_vectors

print("\nStep 4: Building GraphSAGE Model...")

class GraphSAGE(nn.Module):
    def __init__(self, num_nodes, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        
    def encode(self, node_ids, edge_index, edge_weight=None):
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
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 5: Training (10 epochs with edge batching)...")

batch_size = 10000
num_edges = data.num_edges
num_batches = (num_edges + batch_size - 1) // batch_size

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    perm = torch.randperm(num_edges)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_edges)
        batch_edges = perm[start_idx:end_idx]
        
        batch_edge_index = data.edge_index[:, batch_edges]
        batch_edge_attr = data.edge_attr[batch_edges]
        
        unique_nodes = torch.unique(batch_edge_index)
        num_unique = len(unique_nodes)
        
        local_ids = torch.arange(num_unique)
        
        node_to_local = torch.zeros(data.num_nodes, dtype=torch.long)
        node_to_local[unique_nodes] = local_ids
        
        local_edge_index = torch.stack([
            node_to_local[batch_edge_index[0]],
            node_to_local[batch_edge_index[1]]
        ])
        
        optimizer.zero_grad()
        
        z = model.encode(unique_nodes, local_edge_index, batch_edge_attr)
        
        src, dst = local_edge_index
        reconstructed = (z[src] * z[dst]).sum(dim=1)
        
        loss = F.mse_loss(reconstructed, batch_edge_attr)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

print("\nStep 6: Extracting provider embeddings...")

model.eval()
provider_embeddings_list = []

chunk_size = 2000
num_chunks = (n_providers + chunk_size - 1) // chunk_size

with torch.no_grad():
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_providers)
        
        mask = ((data.edge_index[0] >= start_idx) & (data.edge_index[0] < end_idx)) | \
               ((data.edge_index[1] >= start_idx) & (data.edge_index[1] < end_idx))
        
        chunk_edge_index = data.edge_index[:, mask]
        chunk_edge_attr = data.edge_attr[mask]
        
        all_nodes = torch.unique(chunk_edge_index)
        num_nodes = len(all_nodes)
        
        local_ids = torch.arange(num_nodes)
        
        node_to_local = torch.zeros(data.num_nodes, dtype=torch.long)
        node_to_local[all_nodes] = local_ids
        
        local_edge_index = torch.stack([
            node_to_local[chunk_edge_index[0]],
            node_to_local[chunk_edge_index[1]]
        ])
        
        z = model.encode(all_nodes, local_edge_index, chunk_edge_attr)
        
        target_mask = (all_nodes >= start_idx) & (all_nodes < end_idx)
        chunk_embeddings = z[target_mask].cpu().numpy()
        
        provider_embeddings_list.append(chunk_embeddings)
        
        if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
            print(f"  Processed {end_idx}/{n_providers} providers")

provider_embeddings = np.vstack(provider_embeddings_list)

provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = pin_list

print(f"\nEmbeddings shape: {embeddings_df.shape}")
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
print("\n" + "="*80)
print("✓ CPU-ONLY VERSION - NO GPU DEPENDENCIES")
print("="*80)
