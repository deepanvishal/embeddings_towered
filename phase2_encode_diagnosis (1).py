import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

diag_matrix = load_npz('diagnosis_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('specialty_diagnosis_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

with open('specialty_diagnosis_autoencoders_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']
trained_specialties = metadata['trained_specialties']
LATENT_DIM = metadata['latent_dim']

print(f"Diagnosis matrix: {diag_matrix.shape}")
print(f"Total hospitals: {len(all_pins)}")
print(f"Trained specialties: {len(trained_specialties)}")
print(f"Latent dim per specialty: {LATENT_DIM}")

# Convert to tensor
print("\nConverting to tensor...")
diag_tensor = torch.FloatTensor(diag_matrix.toarray()).to(device)
print(f"Diagnosis tensor: {diag_tensor.shape}")

# ============================================================================
# LOAD AUTOENCODER ARCHITECTURE
# ============================================================================

class SpecialtyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================

print("\n" + "="*80)
print("LOADING TRAINED DIAGNOSIS SPECIALTY AUTOENCODERS")
print("="*80)

autoencoders = {}

for specialty in trained_specialties:
    model_filename = f"autoencoder_diag_{specialty.replace(' ', '_')}.pth"
    
    input_dim = len(specialty_code_indices[specialty])
    
    model = SpecialtyAutoencoder(input_dim, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    
    autoencoders[specialty] = model
    print(f"✓ Loaded {specialty:25s} (input: {input_dim:3d}, latent: {LATENT_DIM})")

print(f"\nLoaded {len(autoencoders)} autoencoders")

# ============================================================================
# ENCODE ALL HOSPITALS
# ============================================================================

print("\n" + "="*80)
print("ENCODING ALL HOSPITALS WITH DIAGNOSIS AUTOENCODERS")
print("="*80)

n_hospitals = diag_tensor.shape[0]
n_specialties = len(trained_specialties)
total_embedding_dim = n_specialties * LATENT_DIM

print(f"\nEncoding {n_hospitals} hospitals...")
print(f"Each hospital → {n_specialties} specialties × {LATENT_DIM} dims = {total_embedding_dim} dims")

all_embeddings = []
specialty_embeddings = {spec: [] for spec in trained_specialties}

batch_size = 256

with torch.no_grad():
    for i in range(0, n_hospitals, batch_size):
        end_idx = min(i + batch_size, n_hospitals)
        batch_embeddings = []
        
        for specialty in trained_specialties:
            code_indices = specialty_code_indices[specialty]
            X_batch = diag_tensor[i:end_idx][:, code_indices]
            
            z = autoencoders[specialty].encode(X_batch)
            
            batch_embeddings.append(z.cpu().numpy())
            specialty_embeddings[specialty].append(z.cpu().numpy())
        
        combined = np.concatenate(batch_embeddings, axis=1)
        all_embeddings.append(combined)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_hospitals} hospitals...")

final_embeddings = np.vstack(all_embeddings)

for specialty in trained_specialties:
    specialty_embeddings[specialty] = np.vstack(specialty_embeddings[specialty])

print(f"\n✓ Encoding complete!")
print(f"Final embedding shape: {final_embeddings.shape}")

# ============================================================================
# SAVE EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("SAVING DIAGNOSIS EMBEDDINGS")
print("="*80)

np.save('hospital_diagnosis_embeddings_480d.npy', final_embeddings)
print(f"✓ Saved: hospital_diagnosis_embeddings_480d.npy ({final_embeddings.shape})")

for specialty, embeddings in specialty_embeddings.items():
    filename = f"diagnosis_embeddings_{specialty.replace(' ', '_')}.npy"
    np.save(filename, embeddings)
    print(f"  Saved: {filename} ({embeddings.shape})")

encoding_metadata = {
    'n_hospitals': n_hospitals,
    'n_specialties': n_specialties,
    'latent_dim_per_specialty': LATENT_DIM,
    'total_embedding_dim': total_embedding_dim,
    'specialty_order': trained_specialties,
    'all_pins': all_pins
}

with open('phase2_diagnosis_encoding_metadata.pkl', 'wb') as f:
    pickle.dump(encoding_metadata, f)

print(f"\n✓ Saved: phase2_diagnosis_encoding_metadata.pkl")

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

print("\n" + "="*80)
print("CREATING DIAGNOSIS EMBEDDINGS DATAFRAME")
print("="*80)

embedding_data = {'PIN': all_pins}

start_idx = 0
for specialty_idx, specialty in enumerate(trained_specialties):
    for dim in range(LATENT_DIM):
        col_name = f'specialty{specialty_idx+1}_{specialty.replace(" ", "")}_{dim}'
        embedding_data[col_name] = final_embeddings[:, start_idx + dim]
    start_idx += LATENT_DIM

embeddings_df = pd.DataFrame(embedding_data)
embeddings_df.to_parquet('hospital_diagnosis_embeddings_480d.parquet', index=False)

print(f"✓ Saved: hospital_diagnosis_embeddings_480d.parquet")
print(f"  Shape: {embeddings_df.shape}")

print("\n" + "="*80)
print("DIAGNOSIS PHASE 2 COMPLETE")
print("="*80)
print("\nNext: Run phase3_compress_diagnosis.py")
