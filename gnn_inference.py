import torch
import torch.nn as nn
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model and mappings...")
checkpoint = torch.load('treatment_gnn_step1.pt')
provider_to_idx = checkpoint['provider_to_idx']
code_to_idx = checkpoint['code_to_idx']
n_providers = checkpoint['n_providers']
n_codes = checkpoint['n_codes']

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

    def forward(self, provider_code_edges, code_provider_edges, edge_weights):
        provider_emb = self.provider_embedding.weight
        code_emb = self.code_embedding.weight
        
        provider_messages = torch.zeros(len(provider_emb), provider_emb.size(1), device=device)
        provider_messages.index_add_(0, code_provider_edges[1], 
                                    code_emb[code_provider_edges[0]] * edge_weights.unsqueeze(1))
        provider_degree = torch.zeros(len(provider_emb), device=device)
        provider_degree.index_add_(0, code_provider_edges[1], edge_weights)
        provider_messages = provider_messages / (provider_degree.unsqueeze(1) + 1e-8)
        
        code_messages = torch.zeros(len(code_emb), code_emb.size(1), device=device)
        code_messages.index_add_(0, provider_code_edges[1], 
                                provider_emb[provider_code_edges[0]] * edge_weights.unsqueeze(1))
        code_degree = torch.zeros(len(code_emb), device=device)
        code_degree.index_add_(0, provider_code_edges[1], edge_weights)
        code_messages = code_messages / (code_degree.unsqueeze(1) + 1e-8)
        
        provider_out = self.provider_mlp(torch.cat([provider_emb, provider_messages], dim=1))
        code_out = self.code_mlp(torch.cat([code_emb, code_messages], dim=1))
        
        return provider_out, code_out

model = TreatmentGNN(n_providers, n_codes).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print("Model loaded.")

def infer_embeddings(new_amt_smry_df):
    grouped = new_amt_smry_df.groupby(['PIN', 'Medical_code'], as_index=False)['claims'].sum()
    
    # Filter to known providers and codes
    grouped = grouped[grouped['PIN'].isin(provider_to_idx.keys())]
    grouped = grouped[grouped['Medical_code'].isin(code_to_idx.keys())]
    
    if len(grouped) == 0:
        print("No known providers/codes in new data")
        return None
    
    # Compute weights
    provider_totals = grouped.groupby('PIN')['claims'].transform('sum')
    grouped['distribution'] = grouped['claims'] / provider_totals
    
    code_provider_counts = grouped.groupby('Medical_code')['PIN'].nunique()
    idf_dict = np.log(len(provider_to_idx) / code_provider_counts).to_dict()
    grouped['idf'] = grouped['Medical_code'].map(idf_dict)
    grouped['weight'] = grouped['distribution'] * (1.0 + grouped['idf'])
    grouped['weight'] = grouped['weight'] / grouped['weight'].max()
    
    # Map indices
    grouped['provider_idx'] = grouped['PIN'].map(provider_to_idx)
    grouped['code_idx'] = grouped['Medical_code'].map(code_to_idx)
    
    provider_indices = grouped['provider_idx'].values.astype(np.int64)
    code_indices = grouped['code_idx'].values.astype(np.int64)
    edge_weights = grouped['weight'].values.astype(np.float32)
    
    provider_code_edges = torch.from_numpy(np.stack([provider_indices, code_indices])).to(device)
    code_provider_edges = torch.stack([provider_code_edges[1], provider_code_edges[0]])
    edge_weights_tensor = torch.from_numpy(edge_weights).to(device)
    
    with torch.no_grad():
        embeddings, _ = model(provider_code_edges, code_provider_edges, edge_weights_tensor)
        embeddings = embeddings.cpu().numpy()
    
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    unique_providers = grouped['PIN'].unique()
    emb_df = pd.DataFrame(
        embeddings[[provider_to_idx[p] for p in unique_providers]],
        columns=[f'treatment_emb_{i}' for i in range(64)]
    )
    emb_df['PIN'] = unique_providers
    
    return emb_df

# Example usage
new_embeddings = infer_embeddings(new_amt_smry_df)
print(new_embeddings.head())
