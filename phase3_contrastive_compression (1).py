import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import random
from collections import defaultdict

# Device should already be set from Phase 2, but check
try:
    device
    print(f"Using existing device: {device}")
except NameError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# ============================================================================
# PHASE 3: CONTRASTIVE COMPRESSION (480 → 128)
# ============================================================================
# Run this in the same notebook after Phase 2
# Assumes the following are already in memory:
#   - final_embeddings (numpy array from Phase 2)
#   - all_pins
#   - pin_to_label

print("\n" + "="*80)
print("PHASE 3: CONTRASTIVE COMPRESSION")
print("="*80)

# Verify data is available
try:
    print(f"\n✓ Using existing data from Phase 2:")
    print(f"  Embeddings shape: {final_embeddings.shape}")
    print(f"  Total hospitals: {len(all_pins)}")
    print(f"  Labeled hospitals: {len(pin_to_label)}")
except NameError:
    print("\nERROR: Data not found in memory!")
    print("Please run Phase 2 first, or load data manually:")
    print("  final_embeddings = np.load('hospital_embeddings_480d.npy')")
    print("  all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()")
    print("  with open('pin_to_label.pkl', 'rb') as f:")
    print("      pin_to_label = pickle.load(f)")
    raise

# Create PIN to index mapping
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# Get labeled indices
labeled_indices = [pin_to_idx[pin] for pin in pin_to_label.keys()]
print(f"  Labeled indices: {len(labeled_indices)}")

# Convert to tensor
embeddings_tensor = torch.FloatTensor(final_embeddings).to(device)
print(f"  Embeddings tensor: {embeddings_tensor.shape}")
print(f"  Device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
INPUT_DIM = 480
OUTPUT_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.0005
EPOCHS = 50
BATCH_SIZE = 120  # 8 samples per specialty × 15 specialties
TEMPERATURE = 0.1
CONTRASTIVE_WEIGHT = 1.0

print(f"\nHyperparameters:")
print(f"  Input dim: {INPUT_DIM}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Output dim: {OUTPUT_DIM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Temperature: {TEMPERATURE}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CompressionNetwork(nn.Module):
    """Compress 480 dims to 128 dims"""
    def __init__(self, input_dim=480, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """
    Supervised Contrastive Loss
    Pull same-class together, push different-class apart
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Create label mask (1 if same class, 0 otherwise)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    similarity_matrix = similarity_matrix / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Mask out self-similarity
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss
    loss = -mean_log_prob_pos
    loss = loss.mean()
    
    return loss


# ============================================================================
# CREATE LABEL MAPPING
# ============================================================================

print("\n" + "="*80)
print("PROCESSING LABELS")
print("="*80)

# Get unique labels and create numeric mapping
unique_labels = sorted(set(pin_to_label.values()))
label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
n_classes = len(unique_labels)

print(f"Number of classes: {n_classes}")
for i, label in enumerate(unique_labels):
    count = sum(1 for l in pin_to_label.values() if l == label)
    print(f"  {i}: {label:25s} ({count} hospitals)")

# Create numeric labels for training
pin_to_label_numeric = {pin: label_to_numeric[label] for pin, label in pin_to_label.items()}

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

print("\n" + "="*80)
print("CREATING TRAIN/VAL SPLIT")
print("="*80)

# Group by label
label_to_indices = defaultdict(list)
for pin, label in pin_to_label.items():
    idx = pin_to_idx[pin]
    numeric_label = label_to_numeric[label]
    label_to_indices[numeric_label].append(idx)

train_indices = []
val_indices = []

for label, indices in label_to_indices.items():
    random.shuffle(indices)
    split_point = int(0.8 * len(indices))
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

print(f"Train samples: {len(train_indices)}")
print(f"Val samples: {len(val_indices)}")

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
# VALIDATION FUNCTION
# ============================================================================

def validate(model, val_indices):
    """Compute validation loss"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for i in range(0, len(val_indices), BATCH_SIZE):
            batch_idx = val_indices[i:i+BATCH_SIZE]
            
            # Get embeddings and labels
            emb_batch = embeddings_tensor[batch_idx]
            labels_batch = torch.tensor(
                [pin_to_label_numeric[all_pins[idx]] for idx in batch_idx],
                dtype=torch.long, device=device
            )
            
            # Forward pass
            compressed = model(emb_batch)
            
            # Compute loss
            loss = supervised_contrastive_loss(compressed, labels_batch, TEMPERATURE)
            val_losses.append(loss.item())
    
    model.train()
    return np.mean(val_losses)


# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("TRAINING CONTRASTIVE COMPRESSION")
print("="*80)

# Initialize model
model = CompressionNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_loss = float('inf')
patience = 10
patience_counter = 0
history = {'train': [], 'val': []}

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    
    # Create balanced batches
    balanced_batches = create_balanced_batches(
        train_indices, all_pins, pin_to_label_numeric, BATCH_SIZE
    )
    
    for batch_idx in balanced_batches:
        # Get embeddings and labels
        emb_batch = embeddings_tensor[batch_idx]
        labels_batch = torch.tensor(
            [pin_to_label_numeric[all_pins[idx]] for idx in batch_idx],
            dtype=torch.long, device=device
        )
        
        # Forward pass
        compressed = model(emb_batch)
        
        # Compute loss
        loss = supervised_contrastive_loss(compressed, labels_batch, TEMPERATURE)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    # Validation
    val_loss = validate(model, val_indices)
    scheduler.step()
    
    train_loss = np.mean(epoch_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        torch.save(model.state_dict(), 'compression_model_best.pth')
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
          f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
          f"Best={best_val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
model.load_state_dict(best_model_state)

print(f"\n✓ Training complete!")
print(f"Best validation loss: {best_val_loss:.4f}")

# ============================================================================
# ENCODE ALL HOSPITALS
# ============================================================================

print("\n" + "="*80)
print("ENCODING ALL HOSPITALS WITH TRAINED MODEL")
print("="*80)

model.eval()

n_hospitals = embeddings_tensor.shape[0]
all_embeddings_128d = []

batch_size_inference = 512

with torch.no_grad():
    for i in range(0, n_hospitals, batch_size_inference):
        end_idx = min(i + batch_size_inference, n_hospitals)
        
        emb_batch = embeddings_tensor[i:end_idx]
        compressed = model(emb_batch)
        
        # Normalize for cosine similarity
        compressed_norm = F.normalize(compressed, p=2, dim=1)
        
        all_embeddings_128d.append(compressed_norm.cpu().numpy())
        
        if (i // batch_size_inference) % 10 == 0:
            print(f"  Encoded {end_idx}/{n_hospitals} hospitals...")

final_embeddings = np.vstack(all_embeddings_128d)

print(f"\n✓ Encoding complete!")
print(f"Final embedding shape: {final_embeddings.shape}")

# ============================================================================
# SAVE FINAL EMBEDDINGS
# ============================================================================

print("\n" + "="*80)
print("SAVING FINAL EMBEDDINGS")
print("="*80)

# Save numpy array
np.save('hospital_embeddings_128d.npy', final_embeddings)
print(f"✓ Saved: hospital_embeddings_128d.npy")

# Create dataframe
embedding_data = {'PIN': all_pins}

for i in range(OUTPUT_DIM):
    embedding_data[f'emb_{i}'] = final_embeddings[:, i]

embeddings_df = pd.DataFrame(embedding_data)
embeddings_df.to_parquet('hospital_embeddings_128d.parquet', index=False)

print(f"✓ Saved: hospital_embeddings_128d.parquet")
print(f"  Shape: {embeddings_df.shape}")

# Save metadata
phase3_metadata = {
    'input_dim': INPUT_DIM,
    'output_dim': OUTPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'n_hospitals': n_hospitals,
    'n_labeled': len(labeled_indices),
    'n_classes': n_classes,
    'best_val_loss': best_val_loss,
    'training_history': history,
    'label_mapping': label_to_numeric
}

with open('phase3_compression_metadata.pkl', 'wb') as f:
    pickle.dump(phase3_metadata, f)

print(f"✓ Saved: phase3_compression_metadata.pkl")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("FINAL EMBEDDING STATISTICS")
print("="*80)

print(f"\nAll hospitals:")
print(f"  Shape: {final_embeddings.shape}")
print(f"  Mean: {final_embeddings.mean():.4f}")
print(f"  Std: {final_embeddings.std():.4f}")
print(f"  Min: {final_embeddings.min():.4f}")
print(f"  Max: {final_embeddings.max():.4f}")

# Per-label statistics
print(f"\nPer-label statistics:")
for label_name, numeric_label in sorted(label_to_numeric.items(), key=lambda x: x[1]):
    label_pins = [pin for pin, l in pin_to_label.items() if l == label_name]
    label_indices = [pin_to_idx[pin] for pin in label_pins]
    label_embeddings = final_embeddings[label_indices]
    
    print(f"  {label_name:25s}: n={len(label_indices):3d}, "
          f"mean={label_embeddings.mean():.4f}, std={label_embeddings.std():.4f}")

print("\n" + "="*80)
print("PHASE 3 COMPLETE - ALL PHASES FINISHED")
print("="*80)
print(f"\nFinal output: hospital_embeddings_128d.parquet")
print(f"  Contains {n_hospitals} hospitals × {OUTPUT_DIM} dimensions")
print(f"\nReady for analysis!")
