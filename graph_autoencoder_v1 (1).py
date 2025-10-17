import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from torch_geometric.nn import GCNConv
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
print("GRAPH AUTOENCODER - STEP 1")
print("="*80)
print(f"Device: {device}\n")

print("Step 1: Processing data (same as JMVAE)...")

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

x = torch.eye(n_providers + n_codes, dtype=torch.float32)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(device)

print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

del amt_smry_sparse, amt_smry_normalized, amt_smry_vectors
torch.cuda.empty_cache()

print("\nStep 3: Building Graph Autoencoder...")

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
    def encode(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)
    
    def forward(self, data):
        z = self.encode(data.x, data.edge_index, data.edge_attr)
        return z

model = GraphAutoencoder(
    input_dim=n_providers + n_codes,
    hidden_dim=256,
    embedding_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nStep 4: Training Graph Autoencoder (10 epochs)...")

epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    z = model(data)
    
    reconstructed = model.decode(z, data.edge_index)
    loss = F.mse_loss(reconstructed, data.edge_attr)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

print("\nStep 5: Extracting provider embeddings...")

model.eval()
with torch.no_grad():
    z = model(data)
    provider_embeddings = z[:n_providers].cpu().numpy()

provider_embeddings = provider_embeddings / (np.linalg.norm(provider_embeddings, axis=1, keepdims=True) + 1e-8)

embeddings_df = pd.DataFrame(
    provider_embeddings,
    columns=[f'emb_{i}' for i in range(128)]
)
embeddings_df['PIN'] = pin_list

print(f"\nEmbeddings created: {embeddings_df.shape}")
print(embeddings_df.head())

print("\nStep 6: Quality check...")

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
print(f"(Good: mean ~0.3-0.5, std > 0.1)")

print("\nStep 7: Saving...")

embeddings_df.to_csv('graph_autoencoder_embeddings.csv', index=False)

torch.save({
    'model_state': model.state_dict(),
    'pin_list': pin_list,
    'n_providers': n_providers,
    'n_codes': n_codes,
    'vectorizer': vectorizer
}, 'graph_autoencoder_model.pt')

print("Saved: graph_autoencoder_embeddings.csv, graph_autoencoder_model.pt")
print("="*80)
