import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.sparse import load_npz
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# Load vectors
diag_matrix = load_npz('diagnosis_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_diagnosis_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']
specialty_stats = specialty_mappings['stats']

print(f"Diagnosis matrix: {diag_matrix.shape}")
print(f"Total PINs: {len(all_pins)}")
print(f"Labeled PINs: {len(pin_to_label)}")
print(f"Specialties: {len(specialty_code_indices)}")

# Convert to dense tensor
print("\nConverting to tensor...")
diag_tensor = torch.FloatTensor(diag_matrix.toarray()).to(device)
print(f"Diagnosis tensor: {diag_tensor.shape}")

# Create PIN to index mapping
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
LATENT_DIM = 32
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

print(f"\nHyperparameters:")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SpecialtyAutoencoder(nn.Module):
    """Hierarchical autoencoder for specialty-specific diagnoses"""
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


def masked_mse_loss(recon, x):
    """MSE loss only on non-zero elements"""
    mask = (x > 0).float()
    mse = ((recon - x) ** 2) * mask
    loss = mse.sum() / (mask.sum() + 1e-8)
    return loss


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_specialty_autoencoder(X_train, X_val, specialty_name, input_dim):
    """Train autoencoder for one specialty"""
    
    print(f"\n{'='*80}")
    print(f"Training: {specialty_name}")
    print(f"{'='*80}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples: {X_val.shape[0]}")
    print(f"Input features: {input_dim}")
    
    # Initialize model
    model = SpecialtyAutoencoder(input_dim, LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training data
    train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            recon, z = model(batch_x)
            loss = masked_mse_loss(recon, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val)
            val_loss = masked_mse_loss(val_recon, X_val)
        
        train_loss = np.mean(train_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Best={best_val_loss:.4f}")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history


# ============================================================================
# TRAIN ALL SPECIALTIES
# ============================================================================

print("\n" + "="*80)
print("TRAINING DIAGNOSIS SPECIALTY AUTOENCODERS")
print("="*80)

trained_models = {}
training_histories = {}

for specialty_name, code_indices in specialty_code_indices.items():
    
    specialty_pins = [pin for pin, label in pin_to_label.items() if label == specialty_name]
    
    if len(specialty_pins) < 10:
        print(f"\nSkipping {specialty_name}: only {len(specialty_pins)} hospitals")
        continue
    
    specialty_indices = [pin_to_idx[pin] for pin in specialty_pins]
    
    # Extract specialty-specific diagnosis columns
    X_specialty = diag_tensor[specialty_indices][:, code_indices]
    
    # Train/val split
    n_train = int(0.8 * len(specialty_indices))
    X_train = X_specialty[:n_train]
    X_val = X_specialty[n_train:]
    
    # Train
    model, history = train_specialty_autoencoder(
        X_train, X_val, specialty_name, len(code_indices)
    )
    
    trained_models[specialty_name] = model
    training_histories[specialty_name] = history
    
    # Save model
    model_filename = f"autoencoder_diag_{specialty_name.replace(' ', '_')}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Saved: {model_filename}")


# ============================================================================
# SAVE METADATA
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS AND METADATA")
print("="*80)

metadata = {
    'trained_specialties': list(trained_models.keys()),
    'latent_dim': LATENT_DIM,
    'input_dims': {spec: len(specialty_code_indices[spec]) for spec in trained_models.keys()},
    'training_histories': training_histories
}

with open('specialty_diagnosis_autoencoders_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\nSaved metadata: specialty_diagnosis_autoencoders_metadata.pkl")
print(f"Trained {len(trained_models)} specialty autoencoders")

# Print summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

for specialty, history in training_histories.items():
    final_train = history['train'][-1]
    final_val = history['val'][-1]
    best_val = min(history['val'])
    print(f"{specialty:25s}: Train={final_train:.4f}, Val={final_val:.4f}, Best={best_val:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS PHASE 1 COMPLETE")
print("="*80)
print("\nNext: Run phase2_encode_diagnosis.py")
