import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming you have loaded:
# proc_tensor, diag_tensor, demo_tensor, plc_tensor, cost_ctg_tensor, PIN_smry_tensor
# pin_list, pin_to_label

# Get dimensions
proc_dim = proc_tensor.shape[1]
diag_dim = diag_tensor.shape[1]
demo_dim = demo_tensor.shape[1]
plc_dim = plc_tensor.shape[1]
cost_dim = cost_ctg_tensor.shape[1]
pin_dim = PIN_smry_tensor.shape[1]
n_samples = proc_tensor.shape[0]

print(f"Dataset dimensions:")
print(f"  Procedures: {proc_dim}")
print(f"  Diagnoses: {diag_dim}")
print(f"  Demographics: {demo_dim}")
print(f"  Place: {plc_dim}")
print(f"  Cost: {cost_dim}")
print(f"  PIN: {pin_dim}")
print(f"  Total samples: {n_samples}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
PROC_LATENT_DIM = 128
DIAG_LATENT_DIM = 96
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
EPOCHS = 30
CLASSIFICATION_WEIGHT = 2.0
PROC_RECON_WEIGHT = 1.0
DIAG_RECON_WEIGHT = 1.0

total_latent_dim = PROC_LATENT_DIM + DIAG_LATENT_DIM + demo_dim + plc_dim + cost_dim + pin_dim

print(f"\nLatent dimensions:")
print(f"  Proc Encoder: {PROC_LATENT_DIM}")
print(f"  Diag Encoder: {DIAG_LATENT_DIM}")
print(f"  Demographics: {demo_dim}")
print(f"  Place: {plc_dim}")
print(f"  Cost: {cost_dim}")
print(f"  PIN: {pin_dim}")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# ENCODE LABELS
# ============================================================================
print("\nEncoding labels...")
label_encoder = LabelEncoder()
all_labels = list(pin_to_label.values())
label_encoder.fit(all_labels)
n_classes = len(label_encoder.classes_)

pin_to_label_numeric = {pin: label_encoder.transform([label])[0] 
                        for pin, label in pin_to_label.items()}

print(f"Number of classes: {n_classes}")
for i, label_name in enumerate(label_encoder.classes_):
    count = sum(1 for l in pin_to_label.values() if l == label_name)
    print(f"  {i}: {label_name} ({count} samples)")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
labeled_indices = [i for i, pin in enumerate(pin_list) if pin in pin_to_label]
labeled_labels = [pin_to_label[pin_list[i]] for i in labeled_indices]

label_to_indices = defaultdict(list)
for idx, label in zip(labeled_indices, labeled_labels):
    label_to_indices[label].append(idx)

train_indices = []
val_indices = []

for label, indices in label_to_indices.items():
    random.shuffle(indices)
    split_point = int(0.8 * len(indices))
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

print(f"\nTrain samples: {len(train_indices)}")
print(f"Val samples: {len(val_indices)}")

# ============================================================================
# DETERMINISTIC AUTOENCODER ARCHITECTURE
# ============================================================================

class DeterministicEncoder(nn.Module):
    """Deterministic hierarchical encoder (no probabilistic sampling)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Hierarchical compression: input → 2048 → 512 → 128 → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class DeterministicDecoder(nn.Module):
    """Deterministic hierarchical decoder"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        
        # Hierarchical decompression: latent → 128 → 512 → 2048 → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, output_dim)
        )
    
    def forward(self, z):
        return self.decoder(z)


class DeterministicAutoencoder(nn.Module):
    """Complete autoencoder tower"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = DeterministicEncoder(input_dim, latent_dim)
        self.decoder = DeterministicDecoder(latent_dim, input_dim)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class DeterministicHybridModel(nn.Module):
    """Deterministic hybrid model with classification head"""
    def __init__(self, proc_dim, diag_dim, proc_latent, diag_latent, 
                 demo_dim, plc_dim, cost_dim, pin_dim, n_classes):
        super().__init__()
        
        # Deterministic autoencoder towers
        self.proc_tower = DeterministicAutoencoder(proc_dim, proc_latent)
        self.diag_tower = DeterministicAutoencoder(diag_dim, diag_latent)
        
        # Classification head
        total_dim = proc_latent + diag_latent + demo_dim + plc_dim + cost_dim + pin_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        proc_recon, proc_emb = self.proc_tower(proc)
        diag_recon, diag_emb = self.diag_tower(diag)
        
        return {
            'proc_recon': proc_recon,
            'proc_emb': proc_emb,
            'diag_recon': diag_recon,
            'diag_emb': diag_emb
        }
    
    def get_embeddings(self, proc, diag, demo, plc, cost, pin):
        """Get embeddings (raw for classifier, normalized for similarity)"""
        proc_emb = self.proc_tower.encode(proc)
        diag_emb = self.diag_tower.encode(diag)
        
        # Raw concatenation for classifier
        raw_emb = torch.cat([proc_emb, diag_emb, demo, plc, cost, pin], dim=1)
        
        # Normalized for cosine similarity
        proc_norm = F.normalize(proc_emb, p=2, dim=1)
        diag_norm = F.normalize(diag_emb, p=2, dim=1)
        demo_norm = F.normalize(demo, p=2, dim=1)
        plc_norm = F.normalize(plc, p=2, dim=1)
        cost_norm = F.normalize(cost, p=2, dim=1)
        pin_norm = F.normalize(pin, p=2, dim=1)
        norm_emb = torch.cat([proc_norm, diag_norm, demo_norm, plc_norm, cost_norm, pin_norm], dim=1)
        
        return raw_emb, norm_emb
    
    def classify(self, raw_emb):
        return self.classifier(raw_emb)


# Initialize model
model = DeterministicHybridModel(
    proc_dim, diag_dim, PROC_LATENT_DIM, DIAG_LATENT_DIM,
    demo_dim, plc_dim, cost_dim, pin_dim, n_classes
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print(f"Parameters per labeled sample: {total_params / len(labeled_indices):.1f}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def masked_mse_loss(recon, x):
    """MSE on log-space with masking for non-zero codes"""
    mask = (x > 0).float()
    mse = ((recon - x) ** 2) * mask
    loss = mse.sum() / (mask.sum() + 1e-8)
    return loss


def compute_loss(outputs, proc, diag, labels, logits):
    """Compute total loss (reconstruction + classification)"""
    # Reconstruction losses
    proc_recon_loss = masked_mse_loss(outputs['proc_recon'], proc)
    diag_recon_loss = masked_mse_loss(outputs['diag_recon'], diag)
    
    # Normalize by non-zero counts
    proc_nnz = (proc > 0).sum()
    diag_nnz = (diag > 0).sum()
    total_nnz = proc_nnz + diag_nnz
    
    proc_recon_norm = proc_recon_loss * (proc_nnz / total_nnz) * PROC_RECON_WEIGHT
    diag_recon_norm = diag_recon_loss * (diag_nnz / total_nnz) * DIAG_RECON_WEIGHT
    
    recon_loss = proc_recon_norm + diag_recon_norm
    
    # Classification loss
    if labels is not None and logits is not None:
        class_loss = F.cross_entropy(logits, labels)
        total_loss = recon_loss + CLASSIFICATION_WEIGHT * class_loss
    else:
        class_loss = torch.tensor(0.0)
        total_loss = recon_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'proc_recon': proc_recon_loss.item(),
        'diag_recon': diag_recon_loss.item(),
        'classification': class_loss.item()
    }


# ============================================================================
# CLASS-BALANCED BATCH SAMPLER
# ============================================================================

def create_balanced_batches(indices, pin_list, pin_to_label_numeric, batch_size):
    """Create batches with equal samples per class"""
    label_to_idx = defaultdict(list)
    for idx in indices:
        pin = pin_list[idx]
        if pin in pin_to_label_numeric:
            label = pin_to_label_numeric[pin]
            label_to_idx[label].append(idx)
    
    labels = list(label_to_idx.keys())
    n_classes = len(labels)
    samples_per_class = batch_size // n_classes
    
    for label in labels:
        random.shuffle(label_to_idx[label])
    
    min_size = min(len(label_to_idx[label]) for label in labels)
    max_batches = min_size // samples_per_class
    
    batches = []
    pointers = {label: 0 for label in labels}
    
    for _ in range(max_batches):
        batch = []
        for label in labels:
            idx_list = label_to_idx[label]
            start = pointers[label]
            end = start + samples_per_class
            
            if end > len(idx_list):
                batch.extend(idx_list[start:])
                needed = end - len(idx_list)
                batch.extend(idx_list[:needed])
                pointers[label] = needed
            else:
                batch.extend(idx_list[start:end])
                pointers[label] = end
        
        batches.append(batch)
    
    random.shuffle(batches)
    return batches


# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, val_indices):
    """Validation"""
    model.eval()
    val_losses = []
    
    val_labeled = [idx for idx in val_indices if pin_list[idx] in pin_to_label_numeric]
    if len(val_labeled) == 0:
        return 0.0
    
    with torch.no_grad():
        for i in range(0, len(val_labeled), BATCH_SIZE):
            batch_idx = val_labeled[i:i+BATCH_SIZE]
            
            proc_batch = proc_tensor[batch_idx].to(device)
            diag_batch = diag_tensor[batch_idx].to(device)
            demo_batch = demo_tensor[batch_idx].to(device)
            plc_batch = plc_tensor[batch_idx].to(device)
            cost_batch = cost_ctg_tensor[batch_idx].to(device)
            pin_batch = PIN_smry_tensor[batch_idx].to(device)
            
            labels = torch.tensor(
                [pin_to_label_numeric[pin_list[idx]] for idx in batch_idx],
                dtype=torch.long, device=device
            )
            
            outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            raw_emb, _ = model.get_embeddings(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            logits = model.classify(raw_emb)
            
            loss, _ = compute_loss(outputs, proc_batch, diag_batch, labels, logits)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)


# ============================================================================
# TRAINING
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_loss = float('inf')
history = {'train': [], 'val': []}

print("\n" + "="*80)
print("TRAINING: DETERMINISTIC HIERARCHICAL AUTOENCODER + CLASSIFICATION")
print("="*80)
print("Architecture: Deterministic (no VAE sampling)")
print("Loss: Reconstruction + Classification (no KL divergence)")
print("="*80)

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = {'total': 0, 'proc_recon': 0, 'diag_recon': 0, 'classification': 0}
    
    balanced_batches = create_balanced_batches(train_indices, pin_list, pin_to_label_numeric, BATCH_SIZE)
    
    n_batches = 0
    for batch_idx in balanced_batches:
        proc_batch = proc_tensor[batch_idx].to(device)
        diag_batch = diag_tensor[batch_idx].to(device)
        demo_batch = demo_tensor[batch_idx].to(device)
        plc_batch = plc_tensor[batch_idx].to(device)
        cost_batch = cost_ctg_tensor[batch_idx].to(device)
        pin_batch = PIN_smry_tensor[batch_idx].to(device)
        
        labels = torch.tensor(
            [pin_to_label_numeric[pin_list[idx]] for idx in batch_idx],
            dtype=torch.long, device=device
        )
        
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        raw_emb, _ = model.get_embeddings(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        logits = model.classify(raw_emb)
        
        loss, losses_dict = compute_loss(outputs, proc_batch, diag_batch, labels, logits)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        n_batches += 1
    
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    val_loss = validate(model, val_indices)
    scheduler.step()
    
    history['train'].append(epoch_losses['total'])
    history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_deterministic.pth')
        print(f'  ✓ New best model saved (val_loss: {val_loss:.4f})')
    
    gap = val_loss - epoch_losses['total']
    
    print(f'Epoch {epoch+1}/{EPOCHS}: '
          f'Train={epoch_losses["total"]:.4f}, Val={val_loss:.4f}, Gap={gap:.4f} | '
          f'Proc={epoch_losses["proc_recon"]:.4f}, Diag={epoch_losses["diag_recon"]:.4f}, '
          f'Class={epoch_losses["classification"]:.4f}')
    
    if gap > 1.5:
        print(f'  ⚠ Overfitting detected (gap > 1.5)')

print("\n" + "="*80)
print(f"Training complete. Best val loss: {best_val_loss:.4f}")
print("="*80)

# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

print("\nExtracting embeddings...")
model.load_state_dict(torch.load('best_model_deterministic.pth'))
model.eval()

all_proc = []
all_diag = []
all_demo = []
all_plc = []
all_cost = []
all_pin = []

batch_size_infer = 256

with torch.no_grad():
    for i in range(0, n_samples, batch_size_infer):
        end_idx = min(i + batch_size_infer, n_samples)
        
        proc_batch = proc_tensor[i:end_idx].to(device)
        diag_batch = diag_tensor[i:end_idx].to(device)
        demo_batch = demo_tensor[i:end_idx].to(device)
        plc_batch = plc_tensor[i:end_idx].to(device)
        cost_batch = cost_ctg_tensor[i:end_idx].to(device)
        pin_batch = PIN_smry_tensor[i:end_idx].to(device)
        
        # Get embeddings from encoder towers
        proc_emb = model.proc_tower.encode(proc_batch)
        diag_emb = model.diag_tower.encode(diag_batch)
        
        # Normalize all embeddings
        proc_norm = F.normalize(proc_emb, p=2, dim=1)
        diag_norm = F.normalize(diag_emb, p=2, dim=1)
        demo_norm = F.normalize(demo_batch, p=2, dim=1)
        plc_norm = F.normalize(plc_batch, p=2, dim=1)
        cost_norm = F.normalize(cost_batch, p=2, dim=1)
        pin_norm = F.normalize(pin_batch, p=2, dim=1)
        
        all_proc.append(proc_norm.cpu().numpy())
        all_diag.append(diag_norm.cpu().numpy())
        all_demo.append(demo_norm.cpu().numpy())
        all_plc.append(plc_norm.cpu().numpy())
        all_cost.append(cost_norm.cpu().numpy())
        all_pin.append(pin_norm.cpu().numpy())

# Stack all batches
tower_embeddings = {
    'proc': np.vstack(all_proc),
    'diag': np.vstack(all_diag),
    'demo': np.vstack(all_demo),
    'plc': np.vstack(all_plc),
    'cost': np.vstack(all_cost),
    'pin': np.vstack(all_pin)
}

# Create dataframe with tower naming convention
embedding_data = {'PIN': pin_list}

for i in range(PROC_LATENT_DIM):
    embedding_data[f'tower1_proc_emb_{i}'] = tower_embeddings['proc'][:, i]
for i in range(DIAG_LATENT_DIM):
    embedding_data[f'tower2_diag_emb_{i}'] = tower_embeddings['diag'][:, i]
for i in range(demo_dim):
    embedding_data[f'tower3_demo_emb_{i}'] = tower_embeddings['demo'][:, i]
for i in range(plc_dim):
    embedding_data[f'tower4_plc_emb_{i}'] = tower_embeddings['plc'][:, i]
for i in range(cost_dim):
    embedding_data[f'tower5_cost_emb_{i}'] = tower_embeddings['cost'][:, i]
for i in range(pin_dim):
    embedding_data[f'tower6_pin_emb_{i}'] = tower_embeddings['pin'][:, i]

embeddings_df = pd.DataFrame(embedding_data)
embeddings_df.to_parquet('provider_embeddings_deterministic.parquet', index=False)

print(f"\nSaved embeddings:")
print(f"  Tower 1 (Procedures): tower1_proc_emb_0 to tower1_proc_emb_{PROC_LATENT_DIM-1}")
print(f"  Tower 2 (Diagnoses): tower2_diag_emb_0 to tower2_diag_emb_{DIAG_LATENT_DIM-1}")
print(f"  Tower 3 (Demographics): tower3_demo_emb_0 to tower3_demo_emb_{demo_dim-1}")
print(f"  Tower 4 (Place): tower4_plc_emb_0 to tower4_plc_emb_{plc_dim-1}")
print(f"  Tower 5 (Cost): tower5_cost_emb_0 to tower5_cost_emb_{cost_dim-1}")
print(f"  Tower 6 (PIN): tower6_pin_emb_0 to tower6_pin_emb_{pin_dim-1}")
print(f"  Total columns: {len(embedding_data)}")
print("File: provider_embeddings_deterministic.parquet")
print("\n" + "="*80)
print("COMPLETE - DETERMINISTIC MODEL")
print("="*80)
