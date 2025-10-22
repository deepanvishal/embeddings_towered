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
proc_dim = proc_tensor.shape[1]      # 14978
diag_dim = diag_tensor.shape[1]      # 9116
demo_dim = demo_tensor.shape[1]      # 5
plc_dim = plc_tensor.shape[1]        # 4
cost_dim = cost_ctg_tensor.shape[1]  # 11
pin_dim = PIN_smry_tensor.shape[1]   # 2

n_samples = proc_tensor.shape[0]     # 25347

print(f"Dataset dimensions:")
print(f"  Procedures: {proc_dim} cols")
print(f"  Diagnoses: {diag_dim} cols")
print(f"  Demographics: {demo_dim} cols")
print(f"  Place of Service: {plc_dim} cols")
print(f"  Cost Category: {cost_dim} cols")
print(f"  PIN Summary: {pin_dim} cols")
print(f"  Total samples: {n_samples}")
print(f"  Labeled samples: {len(pin_to_label)}")

# ============================================================================
# IMPROVED HYPERPARAMETERS
# ============================================================================
PROC_LATENT_DIM = 128     # INCREASED from 64 (more capacity)
DIAG_LATENT_DIM = 96      # INCREASED from 48 (more capacity)
LEARNING_RATE = 0.0005
KL_WEIGHT_START = 0.01
KL_WEIGHT_END = 0.05
TRIPLET_MARGIN = 0.5
BATCH_SIZE = 128
EPOCHS = 30
SAVE_EVERY_N_EPOCHS = 5

# Loss balancing weights
PROC_RECON_WEIGHT = 1.0   # Weight for procedure reconstruction
DIAG_RECON_WEIGHT = 1.0   # Weight for diagnosis reconstruction

# Calculate total dimensions
demo_latent_dim = demo_dim
plc_latent_dim = plc_dim
cost_latent_dim = cost_dim
pin_latent_dim = pin_dim

total_latent_dim = PROC_LATENT_DIM + DIAG_LATENT_DIM + demo_latent_dim + plc_latent_dim + cost_latent_dim + pin_latent_dim

print(f"\nLatent dimensions (IMPROVED):")
print(f"  Tower 1 (Procedures): {PROC_LATENT_DIM} [VAE] (increased from 64)")
print(f"  Tower 2 (Diagnoses): {DIAG_LATENT_DIM} [VAE] (increased from 48)")
print(f"  Tower 3 (Demographics): {demo_latent_dim} [Raw]")
print(f"  Tower 4 (Place): {plc_latent_dim} [Raw]")
print(f"  Tower 5 (Cost): {cost_latent_dim} [Raw]")
print(f"  Tower 6 (PIN Summary): {pin_latent_dim} [Raw]")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# ENCODE LABELS TO NUMERIC
# ============================================================================

print("\nEncoding labels to numeric indices...")
label_encoder = LabelEncoder()
all_label_names = list(pin_to_label.values())
label_encoder.fit(all_label_names)

# Convert string labels to numeric indices
pin_to_label_numeric = {pin: label_encoder.transform([label])[0] 
                        for pin, label in pin_to_label.items()}

print(f"\nLabel encoding:")
for i, label_name in enumerate(label_encoder.classes_):
    count = sum(1 for l in pin_to_label.values() if l == label_name)
    print(f"  {i}: {label_name} ({count} samples)")

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

# Get labeled indices
labeled_indices = [i for i, pin in enumerate(pin_list) if pin in pin_to_label]
labeled_labels = [pin_to_label[pin_list[i]] for i in labeled_indices]

# Stratified split by label
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

print(f"\nData split:")
print(f"  Train samples: {len(train_indices)}")
print(f"  Validation samples: {len(val_indices)}")

# ============================================================================
# HIERARCHICAL VAE (NO SKIP CONNECTIONS)
# ============================================================================

class HierarchicalTowerVAE(nn.Module):
    """
    Hierarchical VAE with multi-stage compression
    - Stage 1: input → 2048 (code co-occurrence patterns)
    - Stage 2: 2048 → 512 (procedure groups)
    - Stage 3: 512 → 128 (specialty patterns)
    - Stage 4: 128 → latent (final compression)
    - NO skip connections (lower overfitting risk)
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Hierarchical encoder
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.enc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.enc3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent layers
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        
        # Hierarchical decoder (mirror of encoder)
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.dec2 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.dec3 = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.dec4 = nn.Linear(2048, input_dim)
    
    def encode(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        
        mu = self.mu_layer(h3)
        logvar = self.logvar_layer(h3)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = self.dec1(z)
        h2 = self.dec2(h1)
        h3 = self.dec3(h2)
        recon = self.dec4(h3)
        return recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class HierarchicalHybridModel(nn.Module):
    """Hybrid model with hierarchical VAE towers"""
    def __init__(self, proc_dim, diag_dim, proc_latent, diag_latent):
        super().__init__()
        
        self.proc_tower = HierarchicalTowerVAE(proc_dim, proc_latent)
        self.diag_tower = HierarchicalTowerVAE(diag_dim, diag_latent)
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        proc_recon, proc_mu, proc_logvar = self.proc_tower(proc)
        diag_recon, diag_mu, diag_logvar = self.diag_tower(diag)
        
        return {
            'proc': (proc_recon, proc_mu, proc_logvar),
            'diag': (diag_recon, diag_mu, diag_logvar),
            'demo': demo,
            'plc': plc,
            'cost': cost,
            'pin': pin
        }
    
    def get_embeddings(self, proc, diag, demo, plc, cost, pin, add_noise=False):
        proc_mu, _ = self.proc_tower.encode(proc)
        diag_mu, _ = self.diag_tower.encode(diag)
        
        if add_noise and self.training:
            demo_emb = demo + torch.randn_like(demo) * 0.01
            plc_emb = plc + torch.randn_like(plc) * 0.01
            cost_emb = cost + torch.randn_like(cost) * 0.01
            pin_emb = pin + torch.randn_like(pin) * 0.01
        else:
            demo_emb = demo
            plc_emb = plc
            cost_emb = cost
            pin_emb = pin
        
        proc_mu_norm = F.normalize(proc_mu, p=2, dim=1)
        diag_mu_norm = F.normalize(diag_mu, p=2, dim=1)
        demo_emb_norm = F.normalize(demo_emb, p=2, dim=1)
        plc_emb_norm = F.normalize(plc_emb, p=2, dim=1)
        cost_emb_norm = F.normalize(cost_emb, p=2, dim=1)
        pin_emb_norm = F.normalize(pin_emb, p=2, dim=1)
        
        combined = torch.cat([
            proc_mu_norm, diag_mu_norm, demo_emb_norm, 
            plc_emb_norm, cost_emb_norm, pin_emb_norm
        ], dim=1)
        
        return combined, {
            'proc': proc_mu_norm,
            'diag': diag_mu_norm,
            'demo': demo_emb_norm,
            'plc': plc_emb_norm,
            'cost': cost_emb_norm,
            'pin': pin_emb_norm
        }


# Initialize model
model = HierarchicalHybridModel(proc_dim, diag_dim, PROC_LATENT_DIM, DIAG_LATENT_DIM).to(device)

vae_params = sum(p.numel() for p in model.proc_tower.parameters()) + sum(p.numel() for p in model.diag_tower.parameters())
print(f"\nModel parameters: {vae_params:,} (VAE only)")
print(f"Parameters per labeled sample: {vae_params / len(labeled_indices):.1f}")

# ============================================================================
# IMPROVED LOSS FUNCTIONS
# ============================================================================

def vae_loss_mse_log(recon, x, mu, logvar, kl_weight=0.001, use_masking=True):
    """
    IMPROVED: MSE loss on log-space with optional masking
    
    Args:
        recon: Reconstructed values (predicted log-space)
        x: Original values (already log-transformed with IDF in your preprocessing)
        mu: Latent mean
        logvar: Latent log-variance
        kl_weight: Weight for KL term
        use_masking: If True, only compute loss on non-zero codes
    """
    if use_masking:
        # Create mask for non-zero elements (focus on codes that exist)
        mask = (x > 0).float()
        
        # MSE only on non-zero entries
        squared_error = (recon - x).pow(2) * mask
        recon_loss = squared_error.sum() / (mask.sum() + 1e-8)
    else:
        # Standard MSE on all elements
        recon_loss = F.mse_loss(recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss, kl_loss


def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    """
    Batch-hard triplet loss: mines hardest positive and negative per anchor
    """
    # Compute pairwise distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    
    # For each anchor, find hardest positive and hardest negative
    losses = []
    
    for i, anchor_label in enumerate(labels):
        # Positive mask (same label, different sample)
        positive_mask = (labels == anchor_label) & (torch.arange(len(labels), device=embeddings.device) != i)
        
        # Negative mask (different label)
        negative_mask = labels != anchor_label
        
        if positive_mask.any() and negative_mask.any():
            # Hardest positive (farthest same-class sample)
            hardest_positive_dist = pairwise_dist[i][positive_mask].max()
            
            # Hardest negative (closest different-class sample)
            hardest_negative_dist = pairwise_dist[i][negative_mask].min()
            
            # Triplet loss
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
            losses.append(loss)
    
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=embeddings.device)


def compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, epoch):
    """
    IMPROVED: Compute balanced total loss with MSE on log-space
    """
    losses_dict = {}
    
    # VAE losses for proc and diag
    proc_recon, proc_mu, proc_logvar = outputs['proc']
    diag_recon, diag_mu, diag_logvar = outputs['diag']
    
    # Use improved MSE loss on log-space with masking
    proc_recon_loss, proc_kl_loss = vae_loss_mse_log(
        proc_recon, inputs['proc'], proc_mu, proc_logvar, 
        kl_weight, use_masking=True
    )
    diag_recon_loss, diag_kl_loss = vae_loss_mse_log(
        diag_recon, inputs['diag'], diag_mu, diag_logvar, 
        kl_weight, use_masking=True
    )
    
    # Normalize by number of non-zero elements (for fair comparison)
    proc_nnz = (inputs['proc'] > 0).sum()
    diag_nnz = (inputs['diag'] > 0).sum()
    
    proc_recon_loss_norm = proc_recon_loss * (proc_nnz / (proc_nnz + diag_nnz))
    diag_recon_loss_norm = diag_recon_loss * (diag_nnz / (proc_nnz + diag_nnz))
    
    # Total VAE loss with separate weighting
    vae_loss = (PROC_RECON_WEIGHT * proc_recon_loss_norm + 
                DIAG_RECON_WEIGHT * diag_recon_loss_norm + 
                kl_weight * (proc_kl_loss + diag_kl_loss))
    
    losses_dict['proc_recon'] = proc_recon_loss.item()
    losses_dict['diag_recon'] = diag_recon_loss.item()
    losses_dict['kl'] = (proc_kl_loss + diag_kl_loss).item()
    
    # Triplet loss
    if batch_labels is not None and len(batch_labels) > 0:
        triplet_loss = batch_hard_triplet_loss(combined_emb, batch_labels, margin=TRIPLET_MARGIN)
        losses_dict['triplet'] = triplet_loss.item()
        
        # Balance triplet with reconstruction
        total_loss = vae_loss + 1.0 * triplet_loss
    else:
        losses_dict['triplet'] = 0.0
        total_loss = vae_loss
    
    losses_dict['total'] = total_loss.item()
    
    return total_loss, losses_dict


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model, val_indices, kl_weight):
    """Compute validation loss"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i in range(0, len(val_indices), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            
            proc_batch = proc_tensor[batch_idx].to(device)
            diag_batch = diag_tensor[batch_idx].to(device)
            demo_batch = demo_tensor[batch_idx].to(device)
            plc_batch = plc_tensor[batch_idx].to(device)
            cost_batch = cost_ctg_tensor[batch_idx].to(device)
            pin_batch = PIN_smry_tensor[batch_idx].to(device)
            
            # Get labels for batch
            batch_labels_list = [pin_to_label_numeric.get(pin_list[idx]) for idx in batch_idx]
            batch_labels = torch.tensor(
                [l for l in batch_labels_list if l is not None], 
                dtype=torch.long, device=device
            ) if any(l is not None for l in batch_labels_list) else None
            
            # Forward pass
            outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            combined_emb, _ = model.get_embeddings(
                proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
            )
            
            # Compute loss
            inputs = {
                'proc': proc_batch,
                'diag': diag_batch,
                'demo': demo_batch,
                'plc': plc_batch,
                'cost': cost_batch,
                'pin': pin_batch
            }
            
            loss, _ = compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, 0)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)


# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_loss = float('inf')
history = {'train': [], 'val': []}

print("\n" + "="*80)
print("TRAINING WITH HIERARCHICAL ARCHITECTURE (NO SKIP CONNECTIONS)")
print("="*80)
print(f"\nImprovements:")
print(f"  - MSE on log-space")
print(f"  - Masked loss (non-zero codes)")
print(f"  - Hierarchical compression: input → 2048 → 512 → 128 → {PROC_LATENT_DIM}/{DIAG_LATENT_DIM}")
print(f"  - Class-balanced batch sampling")
print("="*80)

# ============================================================================
# CLASS-BALANCED BATCH SAMPLER
# ============================================================================

def create_balanced_batches(train_indices, pin_list, pin_to_label_numeric, batch_size):
    label_to_indices = defaultdict(list)
    for idx in train_indices:
        pin = pin_list[idx]
        if pin in pin_to_label_numeric:
            label = pin_to_label_numeric[pin]
            label_to_indices[label].append(idx)
    
    unique_labels = list(label_to_indices.keys())
    n_classes = len(unique_labels)
    samples_per_class = batch_size // n_classes
    
    for label in unique_labels:
        random.shuffle(label_to_indices[label])
    
    batches = []
    class_pointers = {label: 0 for label in unique_labels}
    
    min_class_size = min(len(label_to_indices[label]) for label in unique_labels)
    max_batches = min_class_size // samples_per_class
    
    for _ in range(max_batches):
        batch = []
        for label in unique_labels:
            indices = label_to_indices[label]
            start = class_pointers[label]
            end = start + samples_per_class
            
            if end > len(indices):
                batch.extend(indices[start:])
                needed = end - len(indices)
                batch.extend(indices[:needed])
                class_pointers[label] = needed
            else:
                batch.extend(indices[start:end])
                class_pointers[label] = end
        
        batches.append(batch)
    
    random.shuffle(batches)
    return batches

# ============================================================================
# TRAINING LOOP
# ============================================================================

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = {
        'total': 0, 'proc_recon': 0, 'diag_recon': 0, 
        'kl': 0, 'triplet': 0
    }
    
    kl_weight = KL_WEIGHT_START + (KL_WEIGHT_END - KL_WEIGHT_START) * (epoch / EPOCHS)
    
    balanced_batches = create_balanced_batches(train_indices, pin_list, pin_to_label_numeric, BATCH_SIZE)
    
    n_batches = 0
    for batch_idx in balanced_batches:
        proc_batch = proc_tensor[batch_idx].to(device)
        diag_batch = diag_tensor[batch_idx].to(device)
        demo_batch = demo_tensor[batch_idx].to(device)
        plc_batch = plc_tensor[batch_idx].to(device)
        cost_batch = cost_ctg_tensor[batch_idx].to(device)
        pin_batch = PIN_smry_tensor[batch_idx].to(device)
        
        batch_labels_list = [pin_to_label_numeric.get(pin_list[idx]) for idx in batch_idx]
        batch_labels = torch.tensor(
            [l for l in batch_labels_list if l is not None], 
            dtype=torch.long, device=device
        ) if any(l is not None for l in batch_labels_list) else None
        
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        combined_emb, _ = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch,
            add_noise=True
        )
        
        inputs = {
            'proc': proc_batch,
            'diag': diag_batch,
            'demo': demo_batch,
            'plc': plc_batch,
            'cost': cost_batch,
            'pin': pin_batch
        }
        
        loss, losses_dict = compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, epoch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        n_batches += 1
    
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    val_loss = validate(model, val_indices, kl_weight)
    scheduler.step()
    
    history['train'].append(epoch_losses['total'])
    history['val'].append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_hierarchical.pth')
        print(f'  New best model saved (val_loss: {val_loss:.4f})')
    
    overfitting_gap = val_loss - epoch_losses['total']
    
    print(f'Epoch {epoch+1}/{EPOCHS}: '
          f'Train={epoch_losses["total"]:.4f}, Val={val_loss:.4f}, Gap={overfitting_gap:.4f} | '
          f'Proc={epoch_losses["proc_recon"]:.4f}, Diag={epoch_losses["diag_recon"]:.4f}, '
          f'KL={epoch_losses["kl"]:.4f}, Triplet={epoch_losses["triplet"]:.4f}')
    
    if overfitting_gap > 1.0:
        print(f'  WARNING: Overfitting detected (Gap > 1.0)')
        
        proc_batch = proc_tensor[batch_idx].to(device)
        diag_batch = diag_tensor[batch_idx].to(device)
            end = start + samples_per_class
            
            if end > len(indices):
                batch.extend(indices[start:])
                needed = end - len(indices)
                batch.extend(indices[:needed])
                class_pointers[label] = needed
            else:
                batch.extend(indices[start:end])
                class_pointers[label] = end
        
        batches.append(batch)
    
    random.shuffle(batches)
    return batches

        pin_batch = PIN_smry_tensor[batch_idx].to(device)
        
        # Get labels for batch
        batch_labels_list = [pin_to_label_numeric.get(pin_list[idx]) for idx in batch_idx]
        batch_labels = torch.tensor(
            [l for l in batch_labels_list if l is not None], 
            dtype=torch.long, device=device
        ) if any(l is not None for l in batch_labels_list) else None
        
        # Forward pass
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        combined_emb, _ = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch,
            add_noise=True
        )
        
        # Compute loss
        inputs = {
            'proc': proc_batch,
            'diag': diag_batch,
            'demo': demo_batch,
            'plc': plc_batch,
            'cost': cost_batch,
            'pin': pin_batch
        }
        
        loss, losses_dict = compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, epoch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        n_batches += 1
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    # Validation
    val_loss = validate(model, val_indices, kl_weight)
    
    # Update learning rate
    scheduler.step()
    
    # Track history
    history['train'].append(epoch_losses['total'])
    history['val'].append(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_improved.pth')
        print(f'  ✓ New best model saved (val_loss: {val_loss:.4f})')
    
    # Calculate overfitting gap
    overfitting_gap = val_loss - epoch_losses['total']
    
    # Print progress every epoch
    print(f'Epoch {epoch+1}/{EPOCHS}: '
          f'Train={epoch_losses["total"]:.4f}, Val={val_loss:.4f}, Gap={overfitting_gap:.4f} | '
          f'Proc={epoch_losses["proc_recon"]:.4f}, Diag={epoch_losses["diag_recon"]:.4f}, '
          f'KL={epoch_losses["kl"]:.4f}, Triplet={epoch_losses["triplet"]:.4f}')
    
    if overfitting_gap > 1.0:
        print(f'  ⚠️  Overfitting detected! (Gap > 1.0)')

print("\n" + "="*80)
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*80)

# ============================================================================
# EXTRACT FINAL EMBEDDINGS USING BEST MODEL
# ============================================================================

print("\nLoading best model and extracting embeddings...")

model.load_state_dict(torch.load('best_model_hierarchical.pth'))
model.eval()

all_combined_embeddings = []
all_tower_embeddings = {
    'proc': [], 'diag': [], 'demo': [], 'plc': [], 'cost': [], 'pin': []
}

batch_size_inference = 256
n_batches_inference = (n_samples + batch_size_inference - 1) // batch_size_inference

with torch.no_grad():
    for batch_idx in range(n_batches_inference):
        start_idx = batch_idx * batch_size_inference
        end_idx = min(start_idx + batch_size_inference, n_samples)
        
        proc_batch = proc_tensor[start_idx:end_idx].to(device)
        diag_batch = diag_tensor[start_idx:end_idx].to(device)
        demo_batch = demo_tensor[start_idx:end_idx].to(device)
        plc_batch = plc_tensor[start_idx:end_idx].to(device)
        cost_batch = cost_ctg_tensor[start_idx:end_idx].to(device)
        pin_batch = PIN_smry_tensor[start_idx:end_idx].to(device)
        
        combined_emb, tower_embs = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
        )
        
        all_combined_embeddings.append(combined_emb.cpu().numpy())
        for key in tower_embs:
            all_tower_embeddings[key].append(tower_embs[key].cpu().numpy())

final_embeddings = np.vstack(all_combined_embeddings)
tower_embeddings = {key: np.vstack(val) for key, val in all_tower_embeddings.items()}

print(f"\nFinal embeddings shape: {final_embeddings.shape}")

# ============================================================================
# CREATE EMBEDDINGS DATAFRAME
# ============================================================================

embedding_data = {'PIN': pin_list}

for i in range(PROC_LATENT_DIM):
    embedding_data[f'tower1_proc_emb_{i}'] = tower_embeddings['proc'][:, i]

for i in range(DIAG_LATENT_DIM):
    embedding_data[f'tower2_diag_emb_{i}'] = tower_embeddings['diag'][:, i]

for i in range(demo_latent_dim):
    embedding_data[f'tower3_demo_emb_{i}'] = tower_embeddings['demo'][:, i]

for i in range(plc_latent_dim):
    embedding_data[f'tower4_plc_emb_{i}'] = tower_embeddings['plc'][:, i]

for i in range(cost_latent_dim):
    embedding_data[f'tower5_cost_emb_{i}'] = tower_embeddings['cost'][:, i]

for i in range(pin_latent_dim):
    embedding_data[f'tower6_pin_emb_{i}'] = tower_embeddings['pin'][:, i]

embeddings_df = pd.DataFrame(embedding_data)

print(f"\nEmbeddings dataframe: {embeddings_df.shape}")
print("\nColumn breakdown:")
print(f"  Tower 1 (Procedures): {PROC_LATENT_DIM} dims")
print(f"  Tower 2 (Diagnoses): {DIAG_LATENT_DIM} dims")
print(f"  Tower 3 (Demographics): {demo_latent_dim} dims")
print(f"  Tower 4 (Place): {plc_latent_dim} dims")
print(f"  Tower 5 (Cost): {cost_latent_dim} dims")
print(f"  Tower 6 (PIN Summary): {pin_latent_dim} dims")

embeddings_df.to_parquet('provider_embeddings_hierarchical.parquet', index=False)

print("\nSaved: provider_embeddings_hierarchical.parquet")
print("\n" + "="*80)
print("COMPLETE")
print("="*80)

# =======================================