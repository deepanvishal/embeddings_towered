import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming you have loaded:
# proc_tensor, diag_tensor, demo_tensor, plc_tensor, cost_ctg_tensor, PIN_smry_tensor
# all_pins, pin_to_label

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

# Latent dimensions per tower
proc_latent_dim = 128
diag_latent_dim = 96
demo_latent_dim = 8
plc_latent_dim = 4
cost_latent_dim = 8
pin_latent_dim = 4

total_latent_dim = proc_latent_dim + diag_latent_dim + demo_latent_dim + plc_latent_dim + cost_latent_dim + pin_latent_dim

print(f"\nLatent dimensions:")
print(f"  Tower 1 (Procedures): {proc_latent_dim} [VAE]")
print(f"  Tower 2 (Diagnoses): {diag_latent_dim} [VAE]")
print(f"  Tower 3 (Demographics): {demo_latent_dim} [Linear]")
print(f"  Tower 4 (Place): {plc_latent_dim} [Linear]")
print(f"  Tower 5 (Cost): {cost_latent_dim} [Linear]")
print(f"  Tower 6 (PIN Summary): {pin_latent_dim} [Linear]")
print(f"  Total: {total_latent_dim}")

# ============================================================================
# MODEL ARCHITECTURE - HYBRID VAE + LINEAR
# ============================================================================

class TowerVAE(nn.Module):
    """VAE tower for complex sparse data"""
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


class HybridMultiTowerModel(nn.Module):
    """Hybrid model: VAE for complex data, Linear for simple data"""
    def __init__(self, proc_dim, diag_dim, demo_dim, plc_dim, cost_dim, pin_dim,
                 proc_latent, diag_latent, demo_latent, plc_latent, cost_latent, pin_latent):
        super().__init__()
        
        # VAE Towers for complex sparse data
        self.proc_tower = TowerVAE(proc_dim, proc_latent, hidden_dims=[1024, 512, 256])
        self.diag_tower = TowerVAE(diag_dim, diag_latent, hidden_dims=[1024, 512, 256])
        
        # Linear Towers for simple dense data
        self.demo_enc = nn.Linear(demo_dim, demo_latent)
        self.plc_enc = nn.Linear(plc_dim, plc_latent)
        self.cost_enc = nn.Linear(cost_dim, cost_latent)
        self.pin_enc = nn.Linear(pin_dim, pin_latent)
    
    def forward(self, proc, diag, demo, plc, cost, pin):
        # VAE towers
        proc_recon, proc_mu, proc_logvar = self.proc_tower(proc)
        diag_recon, diag_mu, diag_logvar = self.diag_tower(diag)
        
        # Linear towers (no reconstruction)
        demo_emb = self.demo_enc(demo)
        plc_emb = self.plc_enc(plc)
        cost_emb = self.cost_enc(cost)
        pin_emb = self.pin_enc(pin)
        
        return {
            'proc': (proc_recon, proc_mu, proc_logvar),
            'diag': (diag_recon, diag_mu, diag_logvar),
            'demo': demo_emb,
            'plc': plc_emb,
            'cost': cost_emb,
            'pin': pin_emb
        }
    
    def get_embeddings(self, proc, diag, demo, plc, cost, pin):
        """Extract and normalize embeddings from all towers"""
        # VAE towers
        proc_mu, _ = self.proc_tower.encode(proc)
        diag_mu, _ = self.diag_tower.encode(diag)
        
        # Linear towers
        demo_emb = self.demo_enc(demo)
        plc_emb = self.plc_enc(plc)
        cost_emb = self.cost_enc(cost)
        pin_emb = self.pin_enc(pin)
        
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
model = HybridMultiTowerModel(
    proc_dim, diag_dim, demo_dim, plc_dim, cost_dim, pin_dim,
    proc_latent_dim, diag_latent_dim, demo_latent_dim, 
    plc_latent_dim, cost_latent_dim, pin_latent_dim
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def vae_loss(recon, x, mu, logvar, kl_weight=0.001):
    """VAE loss = Reconstruction + KL divergence"""
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss for supervised learning"""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def compute_total_loss(outputs, inputs, use_triplet=False, 
                       triplet_embeddings=None, kl_weight=0.001):
    """Compute combined loss"""
    total_loss = 0
    losses_dict = {}
    
    # VAE losses for proc and diag only
    proc_recon, proc_mu, proc_logvar = outputs['proc']
    diag_recon, diag_mu, diag_logvar = outputs['diag']
    
    proc_loss = vae_loss(proc_recon, inputs['proc'], proc_mu, proc_logvar, kl_weight)
    diag_loss = vae_loss(diag_recon, inputs['diag'], diag_mu, diag_logvar, kl_weight)
    
    total_loss = proc_loss + diag_loss
    losses_dict['proc_vae'] = proc_loss.item()
    losses_dict['diag_vae'] = diag_loss.item()
    
    # Triplet loss on combined embeddings
    if use_triplet and triplet_embeddings is not None:
        anchor_emb, positive_emb, negative_emb = triplet_embeddings
        trip_loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        total_loss += 0.1 * trip_loss
        losses_dict['triplet'] = trip_loss.item()
    else:
        losses_dict['triplet'] = 0.0
    
    return total_loss, losses_dict

# ============================================================================
# PREPARE VISUALIZATION DATA
# ============================================================================

# Get labeled indices and labels
labeled_indices = [i for i, pin in enumerate(all_pins) if pin in pin_to_label]
labeled_labels = [pin_to_label[all_pins[i]] for i in labeled_indices]

print(f"\nVisualization setup:")
print(f"  Labeled samples for t-SNE: {len(labeled_indices)}")
print(f"  Unique labels: {len(set(labeled_labels))}")

# Calculate perplexity
perplexity = min(50, len(labeled_indices) // 3)
print(f"  t-SNE perplexity: {perplexity}")

# Setup for GIF creation
frames = []
save_every_n_epochs = 5

# Color mapping
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

batch_size = 64
epochs = 30
n_batches = (n_samples + batch_size - 1) // batch_size

print(f"\nTraining setup:")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Batches per epoch: {n_batches}")

# ============================================================================
# TRAINING LOOP WITH VISUALIZATION
# ============================================================================

print("\nStarting training with visualization...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_losses = {'proc_vae': 0, 'diag_vae': 0, 'triplet': 0}
    triplet_count = 0
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        proc_batch = proc_tensor[batch_indices].to(device)
        diag_batch = diag_tensor[batch_indices].to(device)
        demo_batch = demo_tensor[batch_indices].to(device)
        plc_batch = plc_tensor[batch_indices].to(device)
        cost_batch = cost_ctg_tensor[batch_indices].to(device)
        pin_batch = PIN_smry_tensor[batch_indices].to(device)
        batch_pins = [all_pins[i] for i in batch_indices]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch)
        
        inputs = {
            'proc': proc_batch,
            'diag': diag_batch
        }
        
        # Triplet loss setup
        use_triplet = False
        triplet_embeddings = None
        
        labeled_in_batch = []
        labels_in_batch = []
        for i, pin in enumerate(batch_pins):
            if pin in pin_to_label:
                labeled_in_batch.append(i)
                labels_in_batch.append(pin_to_label[pin])
        
        if len(labeled_in_batch) >= 2:
            # Extract embeddings for triplet loss
            combined_emb, _ = model.get_embeddings(
                proc_batch, diag_batch, demo_batch, plc_batch, cost_batch, pin_batch
            )
            
            # Create triplets
            triplets_anchor = []
            triplets_positive = []
            triplets_negative = []
            
            for i in range(len(labeled_in_batch)):
                anchor_idx = labeled_in_batch[i]
                anchor_label = labels_in_batch[i]
                
                positive_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch))
                                     if labels_in_batch[j] == anchor_label and j != i]
                
                if positive_candidates:
                    positive_idx = random.choice(positive_candidates)
                    
                    negative_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch))
                                         if labels_in_batch[j] != anchor_label]
                    
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                        
                        triplets_anchor.append(combined_emb[anchor_idx:anchor_idx+1])
                        triplets_positive.append(combined_emb[positive_idx:positive_idx+1])
                        triplets_negative.append(combined_emb[negative_idx:negative_idx+1])
            
            if triplets_anchor:
                use_triplet = True
                triplet_embeddings = (
                    torch.cat(triplets_anchor),
                    torch.cat(triplets_positive),
                    torch.cat(triplets_negative)
                )
                triplet_count += 1
        
        # Compute loss
        loss, losses_dict = compute_total_loss(
            outputs, inputs, use_triplet, triplet_embeddings, kl_weight=0.001
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        epoch_loss += loss.item()
        for key, val in losses_dict.items():
            epoch_losses[key] += val
    
    # Average losses
    avg_loss = epoch_loss / n_batches
    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches if key != 'triplet' else triplet_count, 1)
    
    scheduler.step(avg_loss)
    
    # Print progress
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f'\nEpoch {epoch}/{epochs}:')
        print(f'  Total Loss: {avg_loss:.2f}')
        print(f'  Proc VAE: {epoch_losses["proc_vae"]:.2f}, Diag VAE: {epoch_losses["diag_vae"]:.2f}')
        print(f'  Triplet: {epoch_losses["triplet"]:.4f} ({triplet_count} batches)')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # ========================================================================
    # VISUALIZATION EVERY 5 EPOCHS
    # ========================================================================
    
    if epoch % save_every_n_epochs == 0 or epoch == epochs - 1:
        print(f"  Generating t-SNE visualization for epoch {epoch}...")
        
        model.eval()
        with torch.no_grad():
            # Extract embeddings for LABELED providers only
            labeled_proc = proc_tensor[labeled_indices].to(device)
            labeled_diag = diag_tensor[labeled_indices].to(device)
            labeled_demo = demo_tensor[labeled_indices].to(device)
            labeled_plc = plc_tensor[labeled_indices].to(device)
            labeled_cost = cost_ctg_tensor[labeled_indices].to(device)
            labeled_pin = PIN_smry_tensor[labeled_indices].to(device)
            
            # Get combined normalized embeddings
            combined_emb, _ = model.get_embeddings(
                labeled_proc, labeled_diag, labeled_demo,
                labeled_plc, labeled_cost, labeled_pin
            )
            combined_emb_np = combined_emb.cpu().numpy()
        
        # Apply t-SNE
        print(f"  Running t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embedding_2d = tsne.fit_transform(combined_emb_np)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each label with its color
        for label in unique_labels:
            mask = np.array([l == label for l in labeled_labels])
            ax.scatter(
                embedding_2d[mask, 0], 
                embedding_2d[mask, 1],
                c=[label_to_color[label]], 
                label=label,
                s=50, 
                alpha=0.7, 
                edgecolors='black', 
                linewidth=0.5
            )
        
        ax.set_title(f'Provider Embeddings - Epoch {epoch}', fontsize=18, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=14)
        ax.set_ylabel('t-SNE Component 2', fontsize=14)
        ax.legend(loc='best', fontsize=10, ncol=2 if n_labels > 10 else 1)
        ax.grid(True, alpha=0.3)
        
        # Add epoch info
        ax.text(0.02, 0.98, f'Epoch: {epoch}\nLoss: {avg_loss:.2f}\nTriplet: {epoch_losses["triplet"]:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure to memory for GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        
        # Also save individual frame
        plt.savefig(f'tsne_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved frame for epoch {epoch}")
        
        model.train()

print("\n" + "="*80)
print("Training completed!")
print("="*80)

# ============================================================================
# CREATE GIF
# ============================================================================

print("\nCreating GIF from training frames...")

if frames:
    frames[0].save(
        'training_evolution.gif',
        save_all=True,
        append_images=frames[1:],
        duration=1000,  # 1000ms = 1 second per frame
        loop=0  # Infinite loop
    )
    print(f"GIF saved: training_evolution.gif ({len(frames)} frames)")
else:
    print("No frames were generated!")

# ============================================================================
# EXTRACT FINAL EMBEDDINGS
# ============================================================================

print("\nExtracting final embeddings...")

model.eval()
all_combined_embeddings = []
all_tower_embeddings = {
    'proc': [], 'diag': [], 'demo': [], 'plc': [], 'cost': [], 'pin': []
}

batch_size_inference = 128
n_batches_inference = (n_samples + batch_size_inference - 1) // batch_size_inference

print(f"Processing {n_samples} samples in {n_batches_inference} batches...")

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

# Combine all embeddings
final_embeddings = np.vstack(all_combined_embeddings)
tower_embeddings = {key: np.vstack(val) for key, val in all_tower_embeddings.items()}

print(f"\nFinal embeddings shape: {final_embeddings.shape}")
print(f"Expected: ({n_samples}, {total_latent_dim})")

# Create embeddings dataframe
embedding_data = {'PIN': all_pins}

# Add tower embeddings with proper naming
for i in range(proc_latent_dim):
    embedding_data[f'tower1_proc_emb_{i}'] = tower_embeddings['proc'][:, i]

for i in range(diag_latent_dim):
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
embeddings_df.to_parquet('provider_embeddings_hybrid_vae.parquet', index=False)
torch.save(model.state_dict(), 'hybrid_multi_tower_vae.pth')

print("\nSaved:")
print("  - provider_embeddings_hybrid_vae.parquet")
print("  - hybrid_multi_tower_vae.pth")
print("  - training_evolution.gif")
print(f"  - Individual frames: tsne_epoch_*.png")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("All embeddings are normalized and ready for cosine similarity!")
print(f"Total embedding dimensions: {total_latent_dim}")
print(f"  VAE towers (proc + diag): {proc_latent_dim + diag_latent_dim} dims")
print(f"  Linear towers (demo + plc + cost + pin): {demo_latent_dim + plc_latent_dim + cost_latent_dim + pin_latent_dim} dims")
