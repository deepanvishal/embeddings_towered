import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
from PIL import Image
import io
from collections import defaultdict

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


# At the top, after loading pin_to_label
from sklearn.preprocessing import LabelEncoder

# Create label encoder
label_encoder = LabelEncoder()
all_label_names = list(pin_to_label.values())
label_encoder.fit(all_label_names)

# Convert pin_to_label to use numeric indices
pin_to_label_numeric = {pin: label_encoder.transform([label])[0] for pin, label in pin_to_label.items()}

print(f"Label mapping:")
for i, label_name in enumerate(label_encoder.classes_):
    print(f"  {i}: {label_name}")


# Hyperparameters (now configurable)
PROC_LATENT_DIM = 128
DIAG_LATENT_DIM = 96
LEARNING_RATE = 0.0005
KL_WEIGHT_START = 0.001
KL_WEIGHT_END = 0.01
TRIPLET_MARGIN = 0.5
BATCH_SIZE = 128  # Increased for better triplet diversity
EPOCHS = 30
SAVE_EVERY_N_EPOCHS = 5

# Calculate total dimensions
demo_latent_dim = demo_dim
plc_latent_dim = plc_dim
cost_latent_dim = cost_dim
pin_latent_dim = pin_dim

total_latent_dim = PROC_LATENT_DIM + DIAG_LATENT_DIM + demo_latent_dim + plc_latent_dim + cost_latent_dim + pin_latent_dim

print(f"\nLatent dimensions:")
print(f"  Tower 1 (Procedures): {PROC_LATENT_DIM} [VAE]")
print(f"  Tower 2 (Diagnoses): {DIAG_LATENT_DIM} [VAE]")
print(f"  Tower 3 (Demographics): {demo_latent_dim} [Raw]")
print(f"  Tower 4 (Place): {plc_latent_dim} [Raw]")
print(f"  Tower 5 (Cost): {cost_latent_dim} [Raw]")
print(f"  Tower 6 (PIN Summary): {pin_latent_dim} [Raw]")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

# Get labeled indices
labeled_indices = [i for i, pin in enumerate(pin_list) if pin in pin_to_label]
labeled_labels = [pin_to_label[pin_list[i]] for i in labeled_indices]

# Stratified split by label
from collections import defaultdict
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
# MODEL ARCHITECTURE - IMPROVED VAE + RAW
# ============================================================================

class TowerVAE(nn.Module):
    """VAE tower for complex sparse data with improved loss"""
    def __init__(self, input_dim, latent_dim, hidden_dims=[1024, 512, 256]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent layers
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ImprovedHybridModel(nn.Module):
    """Improved hybrid model with better loss handling"""
    def __init__(self, proc_dim, diag_dim, proc_latent, diag_latent):
        super().__init__()
        
        # VAE Towers for complex sparse data
        self.proc_tower = TowerVAE(proc_dim, proc_latent, hidden_dims=[1024, 512, 256])
        self.diag_tower = TowerVAE(diag_dim, diag_latent, hidden_dims=[1024, 512, 256])
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        # VAE towers
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
        """Extract and normalize embeddings from all towers"""
        # VAE towers
        proc_mu, _ = self.proc_tower.encode(proc)
        diag_mu, _ = self.diag_tower.encode(diag)
        
        # Raw features with optional noise for regularization
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
        
        # Normalize each tower independently
        proc_mu_norm = F.normalize(proc_mu, p=2, dim=1)
        diag_mu_norm = F.normalize(diag_mu, p=2, dim=1)
        demo_emb_norm = F.normalize(demo_emb, p=2, dim=1)
        plc_emb_norm = F.normalize(plc_emb, p=2, dim=1)
        cost_emb_norm = F.normalize(cost_emb, p=2, dim=1)
        pin_emb_norm = F.normalize(pin_emb, p=2, dim=1)
        
        # Concatenate all normalized embeddings
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
model = ImprovedHybridModel(proc_dim, diag_dim, PROC_LATENT_DIM, DIAG_LATENT_DIM).to(device)

vae_params = sum(p.numel() for p in model.proc_tower.parameters()) + sum(p.numel() for p in model.diag_tower.parameters())
print(f"\nModel parameters: {vae_params:,} (VAE only, raw features have 0 params)")

# ============================================================================
# IMPROVED LOSS FUNCTIONS
# ============================================================================

def vae_loss_sparse(recon, x, mu, logvar, kl_weight=0.001):
    """
    Improved VAE loss for sparse data using Binary Cross-Entropy
    Assumes x is binary or has been binarized (presence/absence of codes)
    """
    # Binarize input (non-zero = 1)
    x_binary = (x > 0).float()
    
    # BCE loss (better for sparse binary data)
    recon_loss = F.binary_cross_entropy_with_logits(recon, x_binary, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss, kl_loss

def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    """
    Batch-hard triplet loss: mines hardest positive and negative per anchor
    More efficient and effective than random sampling
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
    """Compute balanced total loss"""
    losses_dict = {}
    
    # VAE losses for proc and diag
    proc_recon, proc_mu, proc_logvar = outputs['proc']
    diag_recon, diag_mu, diag_logvar = outputs['diag']
    
    proc_recon_loss, proc_kl_loss = vae_loss_sparse(proc_recon, inputs['proc'], proc_mu, proc_logvar, kl_weight)
    diag_recon_loss, diag_kl_loss = vae_loss_sparse(diag_recon, inputs['diag'], diag_mu, diag_logvar, kl_weight)
    
    # Normalize reconstruction losses by input dimensions
    proc_recon_loss_norm = proc_recon_loss / proc_dim
    diag_recon_loss_norm = diag_recon_loss / diag_dim
    
    # Total VAE loss
    vae_loss = proc_recon_loss_norm + diag_recon_loss_norm + kl_weight * (proc_kl_loss + diag_kl_loss)
    
    losses_dict['proc_recon'] = proc_recon_loss_norm.item()
    losses_dict['diag_recon'] = diag_recon_loss_norm.item()
    losses_dict['kl'] = (proc_kl_loss + diag_kl_loss).item()
    
    # Triplet loss (if we have labels in batch)
    if batch_labels is not None and len(batch_labels) > 0:
        triplet_loss = batch_hard_triplet_loss(combined_emb, batch_labels, margin=TRIPLET_MARGIN)
        losses_dict['triplet'] = triplet_loss.item()
        
        # Balance triplet with reconstruction (both roughly same scale now)
        total_loss = vae_loss + 1.0 * triplet_loss  # Equal weight
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
        # Process validation in batches
        for start_idx in range(0, len(val_indices), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(val_indices))
            batch_indices = val_indices[start_idx:end_idx]
            
            proc_batch = proc_tensor[batch_indices].to(device)
            diag_batch = diag_tensor[batch_indices].to(device)
            demo_batch = demo_tensor[batch_indices].to(device)
            plc_batch = plc_tensor[batch_indices].to(device)
            cost_batch = cost_ctg_tensor[batch_indices].to(device)
            pin_batch = PIN_smry_tensor[batch_indices].to(device)
            
            batch_pins = [pin_list[i] for i in batch_indices]
            batch_labels = torch.tensor([pin_to_label_numeric[pin] for pin in batch_pins], device=device)
            
            outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            combined_emb, _ = model.get_embeddings(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
            
            inputs = {'proc': proc_batch, 'diag': diag_batch}
            
            loss, _ = compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, 0)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)

# ============================================================================
# PREPARE VISUALIZATION DATA
# ============================================================================

print(f"\nVisualization setup:")
print(f"  Labeled samples for UMAP: {len(labeled_indices)}")
print(f"  Unique labels: {len(set(labeled_labels))}")

frames = []
umap_reducer = None
xlim, ylim = None, None

unique_labels = sorted(set(labeled_labels))
n_labels = len(unique_labels)

if n_labels <= 10:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
elif n_labels <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
else:
    colors = plt.cm.hsv(np.linspace(0, 1, n_labels))

label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

n_batches = (len(train_indices) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nTraining setup:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batches per epoch: {n_batches}")
print(f"  KL weight annealing: {KL_WEIGHT_START} → {KL_WEIGHT_END}")

best_val_loss = float('inf')

# ============================================================================
# TRAINING LOOP WITH IMPROVEMENTS
# ============================================================================

print("\nStarting improved training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_losses = {'proc_recon': 0, 'diag_recon': 0, 'kl': 0, 'triplet': 0, 'total': 0}
    
    # KL weight annealing
    kl_weight = KL_WEIGHT_START + (KL_WEIGHT_END - KL_WEIGHT_START) * min(epoch / (EPOCHS * 0.5), 1.0)
    
    # Shuffle training indices
    random.shuffle(train_indices)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(train_indices))
        batch_indices = train_indices[start_idx:end_idx]
        
        # Get batch data
        proc_batch = proc_tensor[batch_indices].to(device)
        diag_batch = diag_tensor[batch_indices].to(device)
        demo_batch = demo_tensor[batch_indices].to(device)
        plc_batch = plc_tensor[batch_indices].to(device)
        cost_batch = cost_ctg_tensor[batch_indices].to(device)
        pin_batch = PIN_smry_tensor[batch_indices].to(device)
        
        batch_pins = [pin_list[i] for i in batch_indices]
        batch_labels = torch.tensor([pin_to_label_numeric[pin] for pin in batch_pins], device=device)

        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        combined_emb, _ = model.get_embeddings(
            proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch, add_noise=True
        )
        
        inputs = {'proc': proc_batch, 'diag': diag_batch}
        
        # Compute loss
        loss, losses_dict = compute_total_loss(outputs, inputs, combined_emb, batch_labels, kl_weight, epoch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping and monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        epoch_loss += loss.item()
        for key, val in losses_dict.items():
            epoch_losses[key] += val
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    # Validation
    val_loss = validate(model, val_indices, kl_weight)
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    # Print progress
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f'\nEpoch {epoch}/{EPOCHS}:')
        print(f'  Train Loss: {epoch_losses["total"]:.4f}, Val Loss: {val_loss:.4f}')
        print(f'  Proc Recon: {epoch_losses["proc_recon"]:.4f}, Diag Recon: {epoch_losses["diag_recon"]:.4f}')
        print(f'  KL: {epoch_losses["kl"]:.4f}, Triplet: {epoch_losses["triplet"]:.4f}')
        print(f'  KL Weight: {kl_weight:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # ========================================================================
    # VISUALIZATION EVERY N EPOCHS
    # ========================================================================
    
    if epoch % SAVE_EVERY_N_EPOCHS == 0 or epoch == EPOCHS - 1:
        print(f"  Generating UMAP visualization...")
        
        model.eval()
        with torch.no_grad():
            labeled_proc = proc_tensor[labeled_indices].to(device)
            labeled_diag = diag_tensor[labeled_indices].to(device)
            labeled_demo = demo_tensor[labeled_indices].to(device)
            labeled_plc = plc_tensor[labeled_indices].to(device)
            labeled_cost = cost_ctg_tensor[labeled_indices].to(device)
            labeled_pin = PIN_smry_tensor[labeled_indices].to(device)
            
            combined_emb, _ = model.get_embeddings(
                labeled_proc, labeled_diag, labeled_demo,
                labeled_plc, labeled_cost, labeled_pin
            )
            combined_emb_np = combined_emb.cpu().numpy()
        
        # Apply UMAP
        if epoch == 0:
            umap_reducer = umap.UMAP(
                n_components=2, n_neighbors=15, min_dist=0.1, 
                random_state=42, n_epochs=500
            )
            embedding_2d = umap_reducer.fit_transform(combined_emb_np)
            xlim = (embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1)
            ylim = (embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1)
        else:
            embedding_2d = umap_reducer.transform(combined_emb_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for label in unique_labels:
            mask = np.array([l == label for l in labeled_labels])
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[label_to_color[label]], label=label,
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5
            )
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Provider Embeddings - Epoch {epoch}', fontsize=18, fontweight='bold')
        ax.set_xlabel('UMAP Component 1', fontsize=14)
        ax.set_ylabel('UMAP Component 2', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, ncol=2 if n_labels > 10 else 1)
        ax.grid(True, alpha=0.3)
        
        ax.text(0.02, 0.02, 
                f'Epoch: {epoch}\nTrain: {epoch_losses["total"]:.4f}\nVal: {val_loss:.4f}\nTriplet: {epoch_losses["triplet"]:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save to file instead of keeping in memory
        frame_path = f'umap_epoch_{epoch:03d}.png'
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        # Save low-res for GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        
        plt.close(fig)
        model.train()

print("\n" + "="*80)
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*80)

# ============================================================================
# CREATE GIF
# ============================================================================

print("\nCreating GIF...")

if frames:
    frames[0].save(
        'training_evolution.gif',
        save_all=True,
        append_images=frames[1:],
        duration=1000,
        loop=0
    )
    print(f"GIF saved: training_evolution.gif ({len(frames)} frames)")

# ============================================================================
# EXTRACT FINAL EMBEDDINGS USING BEST MODEL
# ============================================================================

print("\nLoading best model and extracting embeddings...")

model.load_state_dict(torch.load('best_model.pth'))
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

# Create embeddings dataframe
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
print("\nSample embeddings:")
print(embeddings_df.head())

# Save embeddings
embeddings_df.to_parquet('provider_embeddings_improved.parquet', index=False)

print("\nSaved:")
print("  - provider_embeddings_improved.parquet")
print("  - best_model.pth")
print("  - training_evolution.gif")

print("\n" + "="*80)
print("COMPLETE - IMPROVED VERSION")
print("="*80)
print(f"Total embedding dimensions: {total_latent_dim}")
print(f"Best validation loss: {best_val_loss:.4f}")
