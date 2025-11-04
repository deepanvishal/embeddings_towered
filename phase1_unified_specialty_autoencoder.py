import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.sparse import load_npz
from collections import defaultdict
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

proc_matrix = load_npz('procedure_vectors.npz')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('specialty_code_mappings.pkl', 'rb') as f:
    specialty_mappings = pickle.load(f)

specialty_code_indices = specialty_mappings['code_indices']
specialty_stats = specialty_mappings['stats']

print(f"Procedure matrix: {proc_matrix.shape}")
print(f"Total PINs: {len(all_pins)}")
print(f"Labeled PINs: {len(pin_to_label)}")
print(f"Specialties: {len(specialty_code_indices)}")

proc_tensor = torch.FloatTensor(proc_matrix.toarray()).to(device)
print(f"Procedure tensor: {proc_tensor.shape}")

input_dim = proc_tensor.shape[1]
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# ============================================================================
# CREATE SPECIALTY MAPPINGS
# ============================================================================
print("\nCreating specialty mappings...")

unique_specialties = sorted(set(pin_to_label.values()))
specialty_to_id = {spec: idx for idx, spec in enumerate(unique_specialties)}
id_to_specialty = {idx: spec for spec, idx in specialty_to_id.items()}
num_specialties = len(unique_specialties)

specialty_to_indices = defaultdict(list)
specialty_counts = defaultdict(int)

for pin, specialty in pin_to_label.items():
    idx = pin_to_idx[pin]
    spec_id = specialty_to_id[specialty]
    specialty_to_indices[spec_id].append(idx)
    specialty_counts[specialty] += 1

print("\nSpecialty distribution:")
for specialty in sorted(specialty_counts.keys(), key=lambda x: specialty_counts[x], reverse=True):
    print(f"  {specialty:25s}: {specialty_counts[specialty]:4d} samples")

# Identify small specialties (< 100 samples)
small_specialties = [spec for spec, count in specialty_counts.items() if count < 100]
print(f"\nSmall specialties (<100 samples): {len(small_specialties)}")
for spec in small_specialties:
    print(f"  {spec}: {specialty_counts[spec]}")

# ============================================================================
# COMPUTE SPECIALTY WEIGHTS (inverse frequency, capped)
# ============================================================================
print("\nComputing specialty weights...")

total_samples = len(pin_to_label)
specialty_weights = {}

for specialty, count in specialty_counts.items():
    spec_id = specialty_to_id[specialty]
    weight = total_samples / (num_specialties * count)
    specialty_weights[spec_id] = weight

min_weight = min(specialty_weights.values())
max_weight = 10 * min_weight

for spec_id in specialty_weights:
    specialty_weights[spec_id] = min(specialty_weights[spec_id], max_weight)

weight_tensor = torch.FloatTensor([specialty_weights[i] for i in range(num_specialties)]).to(device)

print("\nSpecialty weights (capped at 10x minimum):")
for spec_id in range(num_specialties):
    spec_name = id_to_specialty[spec_id]
    print(f"  {spec_name:25s}: {specialty_weights[spec_id]:.3f}")

# ============================================================================
# PREPARE DATA SPLITS
# ============================================================================
print("\nPreparing train/val/test splits...")

all_indices = list(range(len(all_pins)))
specialty_ids = torch.LongTensor([specialty_to_id[pin_to_label[all_pins[i]]] for i in all_indices]).to(device)

# 70% train, 15% val, 15% test
np.random.seed(42)
np.random.shuffle(all_indices)

n_train = int(0.7 * len(all_indices))
n_val = int(0.15 * len(all_indices))

train_indices = all_indices[:n_train]
val_indices = all_indices[n_train:n_train+n_val]
test_indices = all_indices[n_train+n_val:]

print(f"Train: {len(train_indices)} ({100*len(train_indices)/len(all_indices):.1f}%)")
print(f"Val:   {len(val_indices)} ({100*len(val_indices)/len(all_indices):.1f}%)")
print(f"Test:  {len(test_indices)} ({100*len(test_indices)/len(all_indices):.1f}%)")

# Create Stage 1 dataset (small specialties + 20% large)
small_specialty_ids = [specialty_to_id[spec] for spec in small_specialties]
stage1_indices = []

for spec_id in small_specialty_ids:
    spec_indices = [idx for idx in train_indices if specialty_ids[idx] == spec_id]
    stage1_indices.extend(spec_indices)

large_specialty_ids = [sid for sid in range(num_specialties) if sid not in small_specialty_ids]
for spec_id in large_specialty_ids:
    spec_indices = [idx for idx in train_indices if specialty_ids[idx] == spec_id]
    sample_size = int(0.2 * len(spec_indices))
    stage1_indices.extend(np.random.choice(spec_indices, sample_size, replace=False).tolist())

print(f"\nStage 1 (small specialties + 20% large): {len(stage1_indices)} samples")

# ============================================================================
# BALANCED BATCH SAMPLER
# ============================================================================

class BalancedSpecialtyBatchSampler:
    def __init__(self, indices, specialty_ids_tensor, num_specialties, batch_size):
        self.indices = indices
        self.specialty_ids = specialty_ids_tensor
        self.num_specialties = num_specialties
        self.batch_size = batch_size
        self.samples_per_specialty = max(1, batch_size // num_specialties)
        
        self.specialty_pools = defaultdict(list)
        for idx in indices:
            spec_id = int(self.specialty_ids[idx].item())
            self.specialty_pools[spec_id].append(idx)
        
        for spec_id in self.specialty_pools:
            np.random.shuffle(self.specialty_pools[spec_id])
        
        self.iterators = {spec_id: 0 for spec_id in range(num_specialties)}
    
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = []
        for spec_id in range(self.num_specialties):
            if spec_id not in self.specialty_pools or len(self.specialty_pools[spec_id]) == 0:
                continue
            
            pool = self.specialty_pools[spec_id]
            start = self.iterators[spec_id]
            
            for _ in range(self.samples_per_specialty):
                if start >= len(pool):
                    np.random.shuffle(pool)
                    start = 0
                batch.append(pool[start])
                start += 1
            
            self.iterators[spec_id] = start
        
        if len(batch) == 0:
            raise StopIteration
        
        np.random.shuffle(batch)
        return batch

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SpecialtyConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, num_specialties, latent_dim, specialty_emb_dim=32):
        super().__init__()
        
        self.specialty_embedding = nn.Embedding(num_specialties, specialty_emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + specialty_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + specialty_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        encoder_input = torch.cat([x, spec_emb], dim=1)
        return self.encoder(encoder_input)
    
    def decode(self, z, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        decoder_input = torch.cat([z, spec_emb], dim=1)
        return self.decoder(decoder_input)
    
    def forward(self, x, specialty_id):
        z = self.encode(x, specialty_id)
        recon = self.decode(z, specialty_id)
        return recon, z


def masked_mse_loss(recon, x):
    mask = (x > 0).float()
    mse = ((recon - x) ** 2) * mask
    loss = mse.sum() / (mask.sum() + 1e-8)
    return loss


def weighted_loss(recon, x, specialty_ids, weight_tensor):
    sample_losses = F.mse_loss(recon, x, reduction='none').mean(dim=1)
    batch_weights = weight_tensor[specialty_ids]
    return (sample_losses * batch_weights).mean()

# ============================================================================
# LATENT DIMENSION SEARCH
# ============================================================================
print("\n" + "="*80)
print("LATENT DIMENSION HYPERPARAMETER SEARCH")
print("="*80)

def evaluate_latent_dim(latent_dim, quick_eval=True):
    """
    Evaluate reconstruction quality for a given latent dimension.
    Uses train/val loss ratio to detect overfitting.
    """
    print(f"\nTesting latent_dim={latent_dim}")
    
    model = SpecialtyConditionedAutoencoder(input_dim, num_specialties, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 if quick_eval else 50
    batch_size = 64
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        
        n_batches = len(train_indices) // batch_size
        for i in range(n_batches):
            batch_idx = train_indices[i*batch_size:(i+1)*batch_size]
            batch_x = proc_tensor[batch_idx]
            batch_spec = specialty_ids[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_batch_losses = []
            n_val_batches = len(val_indices) // batch_size
            for i in range(n_val_batches):
                batch_idx = val_indices[i*batch_size:(i+1)*batch_size]
                batch_x = proc_tensor[batch_idx]
                batch_spec = specialty_ids[batch_idx]
                
                recon, z = model(batch_x, batch_spec)
                loss = masked_mse_loss(recon, batch_x)
                val_batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_train_losses)
        val_loss = np.mean(val_batch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    overfit_ratio = final_val / final_train
    
    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val)
    
    # Compute reconstruction variance explained
    model.eval()
    with torch.no_grad():
        sample_idx = val_indices[:500]
        sample_x = proc_tensor[sample_idx]
        sample_spec = specialty_ids[sample_idx]
        recon, z = model(sample_x, sample_spec)
        
        variance_original = torch.var(sample_x).item()
        variance_residual = torch.var(sample_x - recon).item()
        variance_explained = 1 - (variance_residual / variance_original)
    
    print(f"  Final: Train={final_train:.4f}, Val={final_val:.4f}, Ratio={overfit_ratio:.3f}")
    print(f"  Best Val={best_val:.4f} at epoch {best_epoch+1}")
    print(f"  Variance explained: {variance_explained:.3f}")
    
    return {
        'latent_dim': latent_dim,
        'final_train_loss': final_train,
        'final_val_loss': final_val,
        'overfit_ratio': overfit_ratio,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'variance_explained': variance_explained
    }

# Test different latent dimensions
latent_dims = [32, 64, 128, 256]
results = []

for ld in latent_dims:
    result = evaluate_latent_dim(ld, quick_eval=True)
    results.append(result)

print("\n" + "="*80)
print("LATENT DIMENSION SEARCH RESULTS")
print("="*80)
print(f"{'Latent':<8} {'Train':<10} {'Val':<10} {'Ratio':<8} {'VarExp':<8} {'Best Val':<10}")
print("-"*80)
for r in results:
    print(f"{r['latent_dim']:<8} {r['final_train_loss']:<10.4f} {r['final_val_loss']:<10.4f} "
          f"{r['overfit_ratio']:<8.3f} {r['variance_explained']:<8.3f} {r['best_val_loss']:<10.4f}")

# Select best latent dimension (lowest val loss with ratio < 1.5)
valid_results = [r for r in results if r['overfit_ratio'] < 1.5]
if len(valid_results) == 0:
    print("\nWARNING: All configurations show overfitting. Using lowest val loss.")
    best_result = min(results, key=lambda x: x['final_val_loss'])
else:
    best_result = min(valid_results, key=lambda x: x['final_val_loss'])

LATENT_DIM = best_result['latent_dim']
print(f"\nSelected latent_dim: {LATENT_DIM}")
print(f"Rationale: Val loss={best_result['final_val_loss']:.4f}, "
      f"Overfit ratio={best_result['overfit_ratio']:.3f}, "
      f"Variance explained={best_result['variance_explained']:.3f}")

# Save hyperparameter search results
with open('latent_dim_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: latent_dim_search_results.json")

# ============================================================================
# TRAINING SETUP
# ============================================================================
print("\n" + "="*80)
print("TRAINING UNIFIED SPECIALTY-CONDITIONED AUTOENCODER")
print("="*80)

LEARNING_RATE = 0.001
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 70
BATCH_SIZE = 45
PATIENCE = 15

print(f"\nHyperparameters:")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Stage 1 epochs: {STAGE1_EPOCHS}")
print(f"  Stage 2 epochs: {STAGE2_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Early stopping patience: {PATIENCE}")

model = SpecialtyConditionedAutoencoder(input_dim, num_specialties, LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

training_history = {
    'stage1_train': [],
    'stage1_val': [],
    'stage2_train': [],
    'stage2_val': [],
    'per_specialty_val': defaultdict(list)
}

# ============================================================================
# STAGE 1: SMALL SPECIALTY FOCUS
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: FOCUS ON SMALL SPECIALTIES")
print("="*80)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(STAGE1_EPOCHS):
    model.train()
    train_losses = []
    
    np.random.shuffle(stage1_indices)
    n_batches = len(stage1_indices) // BATCH_SIZE
    
    for i in range(n_batches):
        batch_idx = stage1_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = proc_tensor[batch_idx]
        batch_spec = specialty_ids[batch_idx]
        
        recon, z = model(batch_x, batch_spec)
        loss = weighted_loss(recon, batch_x, batch_spec, weight_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_indices) // BATCH_SIZE
        for i in range(n_val_batches):
            batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_x = proc_tensor[batch_idx]
            batch_spec = specialty_ids[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            val_batch_losses.append(loss.item())
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    training_history['stage1_train'].append(train_loss)
    training_history['stage1_val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{STAGE1_EPOCHS}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)
print(f"\nStage 1 complete. Best val loss: {best_val_loss:.4f}")

# ============================================================================
# STAGE 2: BALANCED TRAINING ON ALL SPECIALTIES
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: BALANCED TRAINING ON ALL SPECIALTIES")
print("="*80)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.5)

balanced_sampler = BalancedSpecialtyBatchSampler(
    train_indices, specialty_ids, num_specialties, BATCH_SIZE
)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(STAGE2_EPOCHS):
    model.train()
    train_losses = []
    
    sampler_iter = iter(balanced_sampler)
    for _ in range(len(train_indices) // BATCH_SIZE):
        try:
            batch_idx = next(sampler_iter)
        except StopIteration:
            break
        
        batch_x = proc_tensor[batch_idx]
        batch_spec = specialty_ids[batch_idx]
        
        recon, z = model(batch_x, batch_spec)
        loss = weighted_loss(recon, batch_x, batch_spec, weight_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # Overall validation
    model.eval()
    with torch.no_grad():
        val_batch_losses = []
        n_val_batches = len(val_indices) // BATCH_SIZE
        for i in range(n_val_batches):
            batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            batch_x = proc_tensor[batch_idx]
            batch_spec = specialty_ids[batch_idx]
            
            recon, z = model(batch_x, batch_spec)
            loss = masked_mse_loss(recon, batch_x)
            val_batch_losses.append(loss.item())
        
        # Per-specialty validation
        per_spec_val = {}
        for spec_id in range(num_specialties):
            spec_val_indices = [idx for idx in val_indices if specialty_ids[idx] == spec_id]
            if len(spec_val_indices) > 0:
                spec_x = proc_tensor[spec_val_indices]
                spec_ids = specialty_ids[spec_val_indices]
                recon, z = model(spec_x, spec_ids)
                spec_loss = masked_mse_loss(recon, spec_x).item()
                per_spec_val[spec_id] = spec_loss
                training_history['per_specialty_val'][spec_id].append(spec_loss)
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_batch_losses)
    
    training_history['stage2_train'].append(train_loss)
    training_history['stage2_val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{STAGE2_EPOCHS}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")
        
        # Show per-specialty validation for small specialties
        if (epoch + 1) % 20 == 0:
            print("  Per-specialty val loss (small specialties):")
            for spec in small_specialties:
                spec_id = specialty_to_id[spec]
                if spec_id in per_spec_val:
                    print(f"    {spec:25s}: {per_spec_val[spec_id]:.4f}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_model_state)
print(f"\nStage 2 complete. Best val loss: {best_val_loss:.4f}")

# ============================================================================
# FINAL EVALUATION & OVERFITTING CHECK
# ============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION & OVERFITTING ANALYSIS")
print("="*80)

model.eval()
with torch.no_grad():
    # Train set evaluation
    train_batch_losses = []
    n_train_batches = len(train_indices) // BATCH_SIZE
    for i in range(n_train_batches):
        batch_idx = train_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = proc_tensor[batch_idx]
        batch_spec = specialty_ids[batch_idx]
        recon, z = model(batch_x, batch_spec)
        loss = masked_mse_loss(recon, batch_x)
        train_batch_losses.append(loss.item())
    
    # Val set evaluation
    val_batch_losses = []
    n_val_batches = len(val_indices) // BATCH_SIZE
    for i in range(n_val_batches):
        batch_idx = val_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = proc_tensor[batch_idx]
        batch_spec = specialty_ids[batch_idx]
        recon, z = model(batch_x, batch_spec)
        loss = masked_mse_loss(recon, batch_x)
        val_batch_losses.append(loss.item())
    
    # Test set evaluation
    test_batch_losses = []
    n_test_batches = len(test_indices) // BATCH_SIZE
    for i in range(n_test_batches):
        batch_idx = test_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_x = proc_tensor[batch_idx]
        batch_spec = specialty_ids[batch_idx]
        recon, z = model(batch_x, batch_spec)
        loss = masked_mse_loss(recon, batch_x)
        test_batch_losses.append(loss.item())
    
    final_train = np.mean(train_batch_losses)
    final_val = np.mean(val_batch_losses)
    final_test = np.mean(test_batch_losses)

print(f"\nFinal losses:")
print(f"  Train: {final_train:.4f}")
print(f"  Val:   {final_val:.4f}")
print(f"  Test:  {final_test:.4f}")
print(f"\nOverfitting check:")
print(f"  Val/Train ratio:  {final_val/final_train:.3f} {'[OK]' if final_val/final_train < 1.5 else '[WARNING: Possible overfitting]'}")
print(f"  Test/Train ratio: {final_test/final_train:.3f} {'[OK]' if final_test/final_train < 1.5 else '[WARNING: Possible overfitting]'}")

# Per-specialty final evaluation
print("\nPer-specialty test loss:")
per_specialty_test = {}
with torch.no_grad():
    for spec_id in range(num_specialties):
        spec_test_indices = [idx for idx in test_indices if specialty_ids[idx] == spec_id]
        if len(spec_test_indices) > 5:
            spec_x = proc_tensor[spec_test_indices]
            spec_ids = specialty_ids[spec_test_indices]
            recon, z = model(spec_x, spec_ids)
            spec_loss = masked_mse_loss(recon, spec_x).item()
            per_specialty_test[spec_id] = spec_loss
            spec_name = id_to_specialty[spec_id]
            sample_count = specialty_counts[spec_name]
            print(f"  {spec_name:25s} (n={sample_count:4d}): {spec_loss:.4f}")

# ============================================================================
# SAVE MODEL AND METADATA
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL AND METADATA")
print("="*80)

torch.save(model.state_dict(), 'specialty_conditioned_autoencoder.pth')
print("Saved: specialty_conditioned_autoencoder.pth")

metadata = {
    'model_type': 'SpecialtyConditionedAutoencoder',
    'input_dim': input_dim,
    'latent_dim': LATENT_DIM,
    'num_specialties': num_specialties,
    'specialty_to_id': specialty_to_id,
    'id_to_specialty': id_to_specialty,
    'specialty_counts': dict(specialty_counts),
    'hyperparameters': {
        'learning_rate': LEARNING_RATE,
        'stage1_epochs': STAGE1_EPOCHS,
        'stage2_epochs': STAGE2_EPOCHS,
        'batch_size': BATCH_SIZE,
        'patience': PATIENCE
    },
    'final_performance': {
        'train_loss': final_train,
        'val_loss': final_val,
        'test_loss': final_test,
        'val_train_ratio': final_val / final_train,
        'test_train_ratio': final_test / final_train
    },
    'per_specialty_test_loss': {id_to_specialty[sid]: loss for sid, loss in per_specialty_test.items()},
    'latent_dim_search': results
}

with open('specialty_conditioned_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("Saved: specialty_conditioned_metadata.pkl")

with open('training_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)
print("Saved: training_history.pkl")

# ============================================================================
# USAGE EXAMPLE FOR PHASE 2
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 USAGE EXAMPLE")
print("="*80)

print("\nTo encode providers in Phase 2:")
print("""
# Load model
model = SpecialtyConditionedAutoencoder(input_dim, num_specialties, latent_dim)
model.load_state_dict(torch.load('specialty_conditioned_autoencoder.pth'))
model.eval()

# Load metadata
with open('specialty_conditioned_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

specialty_to_id = metadata['specialty_to_id']

# Encode a provider
provider_data = torch.FloatTensor(provider_vector).unsqueeze(0).to(device)
specialty_id = torch.LongTensor([specialty_to_id[provider_specialty]]).to(device)

with torch.no_grad():
    embedding = model.encode(provider_data, specialty_id)
    # embedding shape: [1, latent_dim]
""")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
