import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Assuming you have loaded:
# proc_tensor, diag_tensor on device
# all_pins, pin_to_label

proc_dim = proc_tensor.shape[1]
diag_dim = diag_tensor.shape[1]
hidden_dim = 256
latent_dim = 128

print(f"Dimensions - Procedures: {proc_dim}, Diagnoses: {diag_dim}")
print(f"Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")

# ============================================================================
# MODEL ARCHITECTURE (DEEPER TOWERS)
# ============================================================================

class TwoModalityEncoder(nn.Module):
    def __init__(self, proc_dim, diag_dim, hidden_dim=256, latent_dim=128):
        super().__init__()
        
        # Procedure encoder (3 layers for complex patterns)
        self.proc_enc = nn.Sequential(
            nn.Linear(proc_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim)
        )
        
        # Diagnosis encoder (3 layers for complex patterns)
        self.diag_enc = nn.Sequential(
            nn.Linear(diag_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim)
        )
        
        # Joint latent layers
        joint_dim = hidden_dim * 2
        self.mu_layer = nn.Linear(joint_dim, latent_dim)
        self.logvar_layer = nn.Linear(joint_dim, latent_dim)
        
        # Decoders
        self.proc_dec = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, proc_dim)
        )
        
        self.diag_dec = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, diag_dim)
        )
    
    def encode(self, proc, diag):
        proc_h = self.proc_enc(proc)
        diag_h = self.diag_enc(diag)
        
        joint_h = torch.cat([proc_h, diag_h], dim=1)
        mu = self.mu_layer(joint_h)
        logvar = self.logvar_layer(joint_h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        proc_recon = self.proc_dec(z)
        diag_recon = self.diag_dec(z)
        
        return proc_recon, diag_recon
    
    def forward(self, proc, diag):
        mu, logvar = self.encode(proc, diag)
        z = self.reparameterize(mu, logvar)
        proc_recon, diag_recon = self.decode(z)
        
        return proc_recon, diag_recon, mu, logvar

model = TwoModalityEncoder(proc_dim, diag_dim, hidden_dim, latent_dim).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def reconstruction_loss(proc_recon, diag_recon, proc_true, diag_true):
    proc_loss = F.mse_loss(proc_recon, proc_true)
    diag_loss = F.mse_loss(diag_recon, diag_true)
    return proc_loss + diag_loss

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ============================================================================
# TRAINING SETUP
# ============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

batch_size = 64
epochs = 30
n_samples = len(all_pins)
n_batches = (n_samples + batch_size - 1) // batch_size

print(f"\nTraining setup:")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Batches per epoch: {n_batches}")
print(f"Total samples: {n_samples}")
print(f"Labeled samples: {len(pin_to_label)}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\nStarting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_triplet_loss = 0
    triplet_batches = 0
    
    indices = torch.randperm(n_samples)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        proc_batch = proc_tensor[batch_indices]
        diag_batch = diag_tensor[batch_indices]
        batch_pins = [all_pins[i] for i in batch_indices]
        
        optimizer.zero_grad()
        
        proc_recon, diag_recon, mu, logvar = model(proc_batch, diag_batch)
        
        recon_loss = reconstruction_loss(proc_recon, diag_recon, proc_batch, diag_batch)
        kl_loss = kl_divergence(mu, logvar) / len(batch_indices)
        
        loss = recon_loss + 0.001 * kl_loss
        
        # Triplet loss
        current_triplet_loss = torch.tensor(0.0).to(device)
        labeled_in_batch = []
        labels_in_batch = []
        
        for i, pin in enumerate(batch_pins):
            if pin in pin_to_label:
                labeled_in_batch.append(i)
                labels_in_batch.append(pin_to_label[pin])
        
        if len(labeled_in_batch) >= 2:
            triplet_count = 0
            
            for i in range(len(labeled_in_batch)):
                anchor_idx = labeled_in_batch[i]
                anchor_label = labels_in_batch[i]
                anchor_embedding = mu[anchor_idx:anchor_idx+1]
                
                positive_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch)) 
                                     if labels_in_batch[j] == anchor_label and j != i]
                
                if positive_candidates:
                    positive_idx = random.choice(positive_candidates)
                    positive_embedding = mu[positive_idx:positive_idx+1]
                    
                    negative_candidates = [labeled_in_batch[j] for j in range(len(labeled_in_batch)) 
                                         if labels_in_batch[j] != anchor_label]
                    
                    if negative_candidates:
                        negative_idx = random.choice(negative_candidates)
                        negative_embedding = mu[negative_idx:negative_idx+1]
                        
                        trip_loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
                        current_triplet_loss += trip_loss
                        triplet_count += 1
            
            if triplet_count > 0:
                current_triplet_loss = current_triplet_loss / triplet_count
                triplet_batches += 1
        
        loss = loss + 0.1 * current_triplet_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_triplet_loss += current_triplet_loss.item()
    
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_kl = total_kl_loss / n_batches
    avg_triplet = total_triplet_loss / max(triplet_batches, 1)
    
    scheduler.step(avg_loss)
    
    if epoch % 5 == 0:
        print(f'Epoch {epoch:2d}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Triplet={avg_triplet:.4f}, LR={optimizer.param_groups[0]["lr"]:.6f}')

print("\nTraining completed!")

# Save model
torch.save(model.state_dict(), 'two_modality_encoder.pth')
print("Model saved to two_modality_encoder.pth")
