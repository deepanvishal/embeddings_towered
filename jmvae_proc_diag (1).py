import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import random
import warnings
warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seeds(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("JMVAE - TWO MODALITIES WITH TRIPLET & CONTRASTIVE LOSS")
print("="*80)
print(f"Device: {device}\n")

# ============================================================================
# SAMPLE DATA GENERATION (REPLACE WITH YOUR REAL DATA)
# ============================================================================
print("Generating sample data...")
np.random.seed(42)

n_pins = 1000
proc_codes = [f'PROC_{i:03d}' for i in range(300)]
diag_codes = [f'DIAG_{i:03d}' for i in range(400)]
pins = list(range(1, n_pins + 1))

amt_data = []
for pin in pins:
    for code in np.random.choice(proc_codes, size=np.random.randint(5, 20), replace=False):
        amt_data.append({'PIN': pin, 'Medical_code': code, 'dollar_value': np.random.exponential(1000), 'code_type': 'Procedure'})
    for code in np.random.choice(diag_codes, size=np.random.randint(3, 15), replace=False):
        amt_data.append({'PIN': pin, 'Medical_code': code, 'dollar_value': np.random.exponential(500), 'code_type': 'Diagnosis'})

amt_smry_df = pd.DataFrame(amt_data)

# Create labels
labels = ['cardiac', 'orthopedic', 'oncology', 'pediatric', 'emergency', 'neurology']
labeled_pins = np.random.choice(pins, size=int(0.3 * n_pins), replace=False)
label_df = pd.DataFrame({'PIN': labeled_pins, 'label': np.random.choice(labels, len(labeled_pins))})

print(f"Amount summary: {amt_smry_df.shape}")
print(f"Labels: {len(label_df)} PINs labeled\n")

# ============================================================================
# SPLIT INTO PROCEDURES AND DIAGNOSES
# ============================================================================
print("Step 1: Splitting data...")

proc_df = amt_smry_df[amt_smry_df['code_type'] == 'Procedure'].copy()
diag_df = amt_smry_df[amt_smry_df['code_type'] == 'Diagnosis'].copy()

# Process procedures
proc_df['dollar_value'] = pd.to_numeric(proc_df['dollar_value'], errors='coerce').astype(np.float32)
proc_grouped = proc_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

proc_dicts = []
proc_pins = []
for pin in proc_grouped['PIN'].unique():
    pin_data = proc_grouped[proc_grouped['PIN'] == pin]
    proc_dicts.append(dict(zip(pin_data['Medical_code'], pin_data['dollar_value'])))
    proc_pins.append(pin)

proc_vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
proc_sparse = proc_vectorizer.fit_transform(proc_dicts)
proc_row_sums = np.array(proc_sparse.sum(axis=1)).flatten()
proc_vectors = proc_sparse.multiply((1.0 / (proc_row_sums + 1e-8)).reshape(-1, 1)).toarray().astype(np.float32)

# Process diagnoses
diag_df['dollar_value'] = pd.to_numeric(diag_df['dollar_value'], errors='coerce').astype(np.float32)
diag_grouped = diag_df.groupby(['PIN', 'Medical_code'])['dollar_value'].sum().reset_index()

diag_dicts = []
diag_pins = []
for pin in diag_grouped['PIN'].unique():
    pin_data = diag_grouped[diag_grouped['PIN'] == pin]
    diag_dicts.append(dict(zip(pin_data['Medical_code'], pin_data['dollar_value'])))
    diag_pins.append(pin)

diag_vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
diag_sparse = diag_vectorizer.fit_transform(diag_dicts)
diag_row_sums = np.array(diag_sparse.sum(axis=1)).flatten()
diag_vectors = diag_sparse.multiply((1.0 / (diag_row_sums + 1e-8)).reshape(-1, 1)).toarray().astype(np.float32)

# Align PINs
common_pins = list(set(proc_pins) & set(diag_pins))
common_pins.sort()

proc_aligned = np.zeros((len(common_pins), proc_vectors.shape[1]), dtype=np.float32)
diag_aligned = np.zeros((len(common_pins), diag_vectors.shape[1]), dtype=np.float32)

proc_pin_to_idx = {pin: idx for idx, pin in enumerate(proc_pins)}
diag_pin_to_idx = {pin: idx for idx, pin in enumerate(diag_pins)}

for i, pin in enumerate(common_pins):
    proc_aligned[i] = proc_vectors[proc_pin_to_idx[pin]]
    diag_aligned[i] = diag_vectors[diag_pin_to_idx[pin]]

proc_tensor = torch.FloatTensor(proc_aligned).to(device)
diag_tensor = torch.FloatTensor(diag_aligned).to(device)

pin_to_label = dict(zip(label_df['PIN'], label_df['label']))
labeled_common = [p for p in common_pins if p in pin_to_label]

print(f"Procedure: {proc_tensor.shape}, Diagnosis: {diag_tensor.shape}")
print(f"Common PINs: {len(common_pins)}, Labeled: {len(labeled_common)}\n")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
print("Step 2: Creating model...")

proc_dim = proc_tensor.shape[1]
diag_dim = diag_tensor.shape[1]
hidden_dim = 128
latent_dim = 64

proc_enc1 = nn.Linear(proc_dim, hidden_dim).to(device)
proc_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
diag_enc1 = nn.Linear(diag_dim, hidden_dim).to(device)
diag_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
mu_layer = nn.Linear(hidden_dim, latent_dim).to(device)
logvar_layer = nn.Linear(hidden_dim, latent_dim).to(device)
proc_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
proc_dec2 = nn.Linear(hidden_dim//2, proc_dim).to(device)
diag_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
diag_dec2 = nn.Linear(hidden_dim//2, diag_dim).to(device)

def encode(proc_batch, diag_batch):
    proc_h = F.relu(proc_enc2(F.relu(proc_enc1(proc_batch))))
    diag_h = F.relu(diag_enc2(F.relu(diag_enc1(diag_batch))))
    joint_h = torch.cat([proc_h, diag_h], dim=1)
    return mu_layer(joint_h), logvar_layer(joint_h)

def decode(z):
    proc_recon = F.softmax(proc_dec2(F.relu(proc_dec1(z))), dim=1)
    diag_recon = F.softmax(diag_dec2(F.relu(diag_dec1(z))), dim=1)
    return proc_recon, diag_recon

def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

def triplet_loss(anchor, positive, negative, margin=1.0):
    return F.relu(F.pairwise_distance(anchor, positive) - F.pairwise_distance(anchor, negative) + margin).mean()

def contrastive_loss(embeddings, labels, temperature=0.5):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    label_mask.fill_diagonal_(0)
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    pos_pairs = torch.clamp(label_mask.sum(dim=1), min=1.0)
    return -(log_prob * label_mask).sum(dim=1) / pos_pairs).mean()

print("Model created\n")

# ============================================================================
# TRAINING VERSION 1: TRIPLET LOSS
# ============================================================================
print("="*80)
print("TRAINING: TRIPLET LOSS")
print("="*80)

all_params = list(proc_enc1.parameters()) + list(proc_enc2.parameters()) + list(diag_enc1.parameters()) + list(diag_enc2.parameters()) + list(mu_layer.parameters()) + list(logvar_layer.parameters()) + list(proc_dec1.parameters()) + list(proc_dec2.parameters()) + list(diag_dec1.parameters()) + list(diag_dec2.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.001)

batch_size = 32
epochs = 50
n_batches = (len(common_pins) + batch_size - 1) // batch_size

for epoch in range(epochs):
    total_loss = 0
    indices = torch.randperm(len(common_pins))
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(common_pins))
        batch_indices = indices[start:end]
        
        proc_batch = proc_tensor[batch_indices]
        diag_batch = diag_tensor[batch_indices]
        batch_pins = [common_pins[i] for i in batch_indices]
        
        optimizer.zero_grad()
        mu, logvar = encode(proc_batch, diag_batch)
        z = reparameterize(mu, logvar)
        proc_recon, diag_recon = decode(z)
        
        recon_loss = F.mse_loss(proc_recon, proc_batch) + F.mse_loss(diag_recon, diag_batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(proc_batch)
        
        trip_loss = torch.tensor(0.0).to(device)
        labeled_idx = [i for i, p in enumerate(batch_pins) if p in pin_to_label]
        labeled_lbl = [pin_to_label[batch_pins[i]] for i in labeled_idx]
        
        if len(labeled_idx) >= 3:
            count = 0
            for i in range(len(labeled_idx)):
                anchor_idx = labeled_idx[i]
                anchor_lbl = labeled_lbl[i]
                pos_cand = [labeled_idx[j] for j in range(len(labeled_idx)) if labeled_lbl[j] == anchor_lbl and j != i]
                neg_cand = [labeled_idx[j] for j in range(len(labeled_idx)) if labeled_lbl[j] != anchor_lbl]
                
                if pos_cand and neg_cand:
                    trip_loss += triplet_loss(mu[anchor_idx:anchor_idx+1], mu[random.choice(pos_cand):random.choice(pos_cand)+1], mu[random.choice(neg_cand):random.choice(neg_cand)+1])
                    count += 1
            
            if count > 0:
                trip_loss = trip_loss / count
        
        loss = recon_loss + 0.001 * kl_loss + 0.1 * trip_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss={total_loss/n_batches:.4f}')

print("Triplet training done!\n")

# Save triplet model
torch.save({
    'proc_enc1': proc_enc1.state_dict(), 'proc_enc2': proc_enc2.state_dict(),
    'diag_enc1': diag_enc1.state_dict(), 'diag_enc2': diag_enc2.state_dict(),
    'mu_layer': mu_layer.state_dict(), 'logvar_layer': logvar_layer.state_dict(),
    'proc_dec1': proc_dec1.state_dict(), 'proc_dec2': proc_dec2.state_dict(),
    'diag_dec1': diag_dec1.state_dict(), 'diag_dec2': diag_dec2.state_dict(),
    'proc_vectorizer': proc_vectorizer, 'diag_vectorizer': diag_vectorizer,
    'common_pins': common_pins, 'proc_dim': proc_dim, 'diag_dim': diag_dim,
    'hidden_dim': hidden_dim, 'latent_dim': latent_dim
}, 'jmvae_triplet_model.pt')

# Extract triplet embeddings
all_emb = []
for i in range(n_batches):
    start = i * batch_size
    end = min(start + batch_size, len(common_pins))
    with torch.no_grad():
        mu, _ = encode(proc_tensor[start:end], diag_tensor[start:end])
        all_emb.append(mu.cpu())

emb_triplet = torch.cat(all_emb).numpy()
pd.DataFrame(emb_triplet, columns=[f'emb_{i}' for i in range(latent_dim)]).assign(PIN=common_pins).to_csv('jmvae_triplet_embeddings.csv', index=False)
print("Saved: jmvae_triplet_model.pt, jmvae_triplet_embeddings.csv\n")

# ============================================================================
# TRAINING VERSION 2: CONTRASTIVE LOSS
# ============================================================================
print("="*80)
print("TRAINING: CONTRASTIVE LOSS")
print("="*80)

# Reinitialize
proc_enc1 = nn.Linear(proc_dim, hidden_dim).to(device)
proc_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
diag_enc1 = nn.Linear(diag_dim, hidden_dim).to(device)
diag_enc2 = nn.Linear(hidden_dim, hidden_dim//2).to(device)
mu_layer = nn.Linear(hidden_dim, latent_dim).to(device)
logvar_layer = nn.Linear(hidden_dim, latent_dim).to(device)
proc_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
proc_dec2 = nn.Linear(hidden_dim//2, proc_dim).to(device)
diag_dec1 = nn.Linear(latent_dim, hidden_dim//2).to(device)
diag_dec2 = nn.Linear(hidden_dim//2, diag_dim).to(device)

all_params = list(proc_enc1.parameters()) + list(proc_enc2.parameters()) + list(diag_enc1.parameters()) + list(diag_enc2.parameters()) + list(mu_layer.parameters()) + list(logvar_layer.parameters()) + list(proc_dec1.parameters()) + list(proc_dec2.parameters()) + list(diag_dec1.parameters()) + list(diag_dec2.parameters())
optimizer = torch.optim.Adam(all_params, lr=0.001)

label_encoder = LabelEncoder()
label_encoder.fit(list(set(pin_to_label.values())))

for epoch in range(epochs):
    total_loss = 0
    indices = torch.randperm(len(common_pins))
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(common_pins))
        batch_indices = indices[start:end]
        
        proc_batch = proc_tensor[batch_indices]
        diag_batch = diag_tensor[batch_indices]
        batch_pins = [common_pins[i] for i in batch_indices]
        
        optimizer.zero_grad()
        mu, logvar = encode(proc_batch, diag_batch)
        z = reparameterize(mu, logvar)
        proc_recon, diag_recon = decode(z)
        
        recon_loss = F.mse_loss(proc_recon, proc_batch) + F.mse_loss(diag_recon, diag_batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(proc_batch)
        
        contrast_loss = torch.tensor(0.0).to(device)
        labeled_idx = [i for i, p in enumerate(batch_pins) if p in pin_to_label]
        
        if len(labeled_idx) >= 2:
            labeled_emb = mu[labeled_idx]
            labeled_lbl = [pin_to_label[batch_pins[i]] for i in labeled_idx]
            labels_enc = torch.LongTensor(label_encoder.transform(labeled_lbl)).to(device)
            contrast_loss = contrastive_loss(labeled_emb, labels_enc)
        
        loss = recon_loss + 0.001 * kl_loss + 0.1 * contrast_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss={total_loss/n_batches:.4f}')

print("Contrastive training done!\n")

# Save contrastive model
torch.save({
    'proc_enc1': proc_enc1.state_dict(), 'proc_enc2': proc_enc2.state_dict(),
    'diag_enc1': diag_enc1.state_dict(), 'diag_enc2': diag_enc2.state_dict(),
    'mu_layer': mu_layer.state_dict(), 'logvar_layer': logvar_layer.state_dict(),
    'proc_dec1': proc_dec1.state_dict(), 'proc_dec2': proc_dec2.state_dict(),
    'diag_dec1': diag_dec1.state_dict(), 'diag_dec2': diag_dec2.state_dict(),
    'proc_vectorizer': proc_vectorizer, 'diag_vectorizer': diag_vectorizer,
    'common_pins': common_pins, 'proc_dim': proc_dim, 'diag_dim': diag_dim,
    'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'label_encoder': label_encoder
}, 'jmvae_contrastive_model.pt')

# Extract contrastive embeddings
all_emb = []
for i in range(n_batches):
    start = i * batch_size
    end = min(start + batch_size, len(common_pins))
    with torch.no_grad():
        mu, _ = encode(proc_tensor[start:end], diag_tensor[start:end])
        all_emb.append(mu.cpu())

emb_contrast = torch.cat(all_emb).numpy()
pd.DataFrame(emb_contrast, columns=[f'emb_{i}' for i in range(latent_dim)]).assign(PIN=common_pins).to_csv('jmvae_contrastive_embeddings.csv', index=False)
print("Saved: jmvae_contrastive_model.pt, jmvae_contrastive_embeddings.csv\n")

print("="*80)
print("COMPLETE! Two models trained:")
print("1. Triplet Loss: jmvae_triplet_model.pt + jmvae_triplet_embeddings.csv")
print("2. Contrastive Loss: jmvae_contrastive_model.pt + jmvae_contrastive_embeddings.csv")
print("="*80)
