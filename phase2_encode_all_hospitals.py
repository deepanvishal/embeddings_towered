import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# Load vectors
proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

with open('specialty_autoencoders_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']
trained_specialties = metadata['trained_specialties']
LATENT_DIM = metadata['latent_dim']

print(f"Procedure matrix: {proc_matrix.shape}")
print(f"Total hospitals: {len(all_pins)}")
print(f"Trained specialties: {len(trained_specialties)}")
print(f"Latent dim per specialty: {LATENT_DIM}")

# Convert to dense tensor
print("\nConverting to tensor...")
proc_tensor = torch.FloatTensor(proc_matrix.toarray()).to(device)
print(f"Procedure tensor: {proc_tensor.shape}")

# ============================================================================
# LOAD AUTOENCODER ARCHITECTURE
# ============================================================================

class SpecialtyAutoencoder(nn.Module):
    """Same architecture as Phase 1"""
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
print("LOADING TRAINED SPECIALTY AUTOENCODERS")
print("="*80)

autoencoders = {}

for specialty in trained_specialties:
    model_filename = f"autoencoder_{specialty.replace(' ', '_')}.pth"
    
    # Get input dimension for this specialty
    input_dim = len(specialty_code_indices[specialty])
    
    # Initialize model
    model = SpecialtyAutoencoder(input_dim, LATENT_DIM).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    
    autoencoders[specialty] = model
    print(f"✓ Loaded {specialty:25s} (input: {input_dim:3d}, latent: {LATENT_DIM})")

print(f"\nLoaded {len(autoencoders)} autoencoders")

# ============================================================================
# ENCODE ALL HOSPITALS
# ============================================================================

print("\n" + "="*80)
print("ENCODING ALL HOSPITALS WITH ALL SPECIALTY AUTOENCODERS")
print("="*80)

n_hospitals = proc_tensor.shape[0]
n_specialties = len(trained_specialties)
total_embedding_dim = n_specialties * LATENT_DIM

print(f"\nEncoding {n_hospitals} hospitals...")
print(f"Each hospital → {n_specialties} specialties × {LATENT_DIM} dims = {total_embedding_dim} dims")

# Storage for embeddings
all_embeddings = []
specialty_embeddings = {spec: [] for spec in trained_specialties}

# Batch processing for efficiency
batch_size = 256

with torch.no_grad():
    for i in range(0, n_hospitals, batch_size):
        end_idx = min(i + batch_size, n_hospitals)
        batch_embeddings = []
        
        # For each specialty
        for specialty in trained_specialties:
            # Get code indices for this specialty
            code_indices = specialty_code_indices[specialty]
            
            # Extract specialty-specific procedures for this batch
            X_batch = proc_tensor[i:end_idx][:, code_indices]
            
            # Encode with specialty autoencoder
            z = autoencoders[specialty].encode(X_batch)
            
            # Store
            batch_embeddings.append(z.cpu().numpy())
            specialty_embeddings[specialty].append(z.cpu().numpy())
        
        # Concatenate all specialty embeddings for this batch
        # Shape: [batch_size, n_specialties * 32]
        combined = np.concatenate(batch_embeddings, axis=1)
        all_embeddings.append(combined)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {end_idx}/{n_hospitals} hospitals...")

# Stack all batches
final_embeddings = np.vstack(all_embeddings)

# Stack specialty-specific embeddings
for specialty in trained_specialties:
    specialty_embeddings[specialty] = np.vstack(specialty_embeddings[specialty])

print(f"\n✓ Encoding complete!")
print(f"Final embedding shape: {final_embeddings.shape}")

# ============================================================================
# SAVE EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("SAVING EMBEDDINGS")
print("="*80)

# Save combined embeddings
np.save('hospital_embeddings_480d.npy', final_embeddings)
print(f"✓ Saved: hospital_embeddings_480d.npy ({final_embeddings.shape})")

# Save specialty-specific embeddings
for specialty, embeddings in specialty_embeddings.items():
    filename = f"embeddings_{specialty.replace(' ', '_')}.npy"
    np.save(filename, embeddings)
    print(f"  Saved: {filename} ({embeddings.shape})")

# Save metadata
encoding_metadata = {
    'n_hospitals': n_hospitals,
    'n_specialties': n_specialties,
    'latent_dim_per_specialty': LATENT_DIM,
    'total_embedding_dim': total_embedding_dim,
    'specialty_order': trained_specialties,
    'all_pins': all_pins
}

with open('phase2_encoding_metadata.pkl', 'wb') as f:
    pickle.dump(encoding_metadata, f)

print(f"\n✓ Saved: phase2_encoding_metadata.pkl")

# ============================================================================
# CREATE EMBEDDINGS DATAFRAME (OPTIONAL)
# ============================================================================

print("\n" + "="*80)
print("CREATING EMBEDDINGS DATAFRAME")
print("="*80)

# Create dataframe with specialty-specific columns
embedding_data = {'PIN': all_pins}

# Add embeddings for each specialty
start_idx = 0
for specialty_idx, specialty in enumerate(trained_specialties):
    for dim in range(LATENT_DIM):
        col_name = f'specialty{specialty_idx+1}_{specialty.replace(" ", "")}_{dim}'
        embedding_data[col_name] = final_embeddings[:, start_idx + dim]
    start_idx += LATENT_DIM

embeddings_df = pd.DataFrame(embedding_data)
embeddings_df.to_parquet('hospital_embeddings_480d.parquet', index=False)

print(f"✓ Saved: hospital_embeddings_480d.parquet")
print(f"  Shape: {embeddings_df.shape}")
print(f"  Columns: PIN + {total_embedding_dim} embedding dimensions")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("ENCODING STATISTICS")
print("="*80)

print(f"\nEmbedding dimensions per specialty:")
for i, specialty in enumerate(trained_specialties):
    start = i * LATENT_DIM
    end = start + LATENT_DIM
    emb = final_embeddings[:, start:end]
    print(f"  {specialty:25s}: dims [{start:3d}:{end:3d}], "
          f"mean={emb.mean():.4f}, std={emb.std():.4f}")

print(f"\nOverall statistics:")
print(f"  Total embedding mean: {final_embeddings.mean():.4f}")
print(f"  Total embedding std: {final_embeddings.std():.4f}")
print(f"  Min value: {final_embeddings.min():.4f}")
print(f"  Max value: {final_embeddings.max():.4f}")

# Check for any NaN or Inf
if np.isnan(final_embeddings).any():
    print(f"  WARNING: Found NaN values!")
if np.isinf(final_embeddings).any():
    print(f"  WARNING: Found Inf values!")

print("\n" + "="*80)
print("PHASE 2 COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  1. hospital_embeddings_480d.npy - Combined embeddings (numpy)")
print(f"  2. hospital_embeddings_480d.parquet - Combined embeddings (pandas)")
print(f"  3. embeddings_[specialty].npy - Per-specialty embeddings")
print(f"  4. phase2_encoding_metadata.pkl - Metadata")
print(f"\nNext step: Run Phase 3 for contrastive compression (480 → 128)")
