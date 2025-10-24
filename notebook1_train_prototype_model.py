"""
NOTEBOOK 1: TRAIN PROTOTYPE-BASED QUERY-DEPENDENT SIMILARITY MODEL
===================================================================

This notebook trains a model that:
1. Learns prototype embeddings for each specialty (15 prototypes)
2. Learns weight profiles for each specialty (6 weights per specialty)
3. Given a query provider, predicts tower weights based on prototype similarity
4. Uses these weights to compute context-dependent similarity

Author: AI Assistant
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import random

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load embeddings
embeddings_df = pd.read_parquet('final_all_towers_278d.parquet')
all_pins = embeddings_df['PIN'].values
embedding_cols = [col for col in embeddings_df.columns if col != 'PIN']
embeddings = embeddings_df[embedding_cols].values

print(f"Embeddings: {embeddings.shape}")
print(f"Total providers: {len(all_pins)}")

# Load labels
with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

print(f"Labeled providers: {len(pin_to_label)}")

# Create mappings
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}
labeled_indices = [pin_to_idx[pin] for pin in pin_to_label.keys()]
labeled_pins = list(pin_to_label.keys())

# Label encoding
unique_labels = sorted(set(pin_to_label.values()))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

n_prototypes = len(unique_labels)
embedding_dim = embeddings.shape[1]

print(f"Number of specialties: {n_prototypes}")
print(f"Embedding dimension: {embedding_dim}")

# Identify tower dimensions
tower_dims = {
    'procedures': (0, 128),
    'diagnoses': (128, 256),
    'demographics': (256, 261),
    'place': (261, 265),
    'cost': (265, 276),
    'pin': (276, 278)
}

print("\nTower structure:")
for tower, (start, end) in tower_dims.items():
    print(f"  {tower:15s}: dims [{start:3d}:{end:3d}] = {end-start:3d} dims")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*80)
print("DEFINING MODEL ARCHITECTURE")
print("="*80)

class PrototypeWeightModel(nn.Module):
    """
    Prototype-based query-dependent similarity model.
    
    For each specialty, learns:
    - A prototype embedding (278 dims)
    - A weight profile (6 tower weights)
    
    At inference:
    - Computes query similarity to all prototypes
    - Blends weight profiles based on prototype similarities
    - Returns tower weights for similarity computation
    """
    
    def __init__(self, n_prototypes, embedding_dim, n_towers=6):
        super().__init__()
        
        self.n_prototypes = n_prototypes
        self.embedding_dim = embedding_dim
        self.n_towers = n_towers
        
        # Learnable prototypes (one per specialty)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim) * 0.1)
        
        # Learnable weight profiles (one per specialty)
        # Each row = tower weights for that specialty
        self.weight_profiles = nn.Parameter(torch.ones(n_prototypes, n_towers))
        
        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, query_emb):
        """
        Args:
            query_emb: [batch, embedding_dim] or [embedding_dim]
        
        Returns:
            weights: [batch, n_towers] or [n_towers] - tower weights
        """
        # Handle single query
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize embeddings
        query_norm = F.normalize(query_emb, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Compute similarity to all prototypes
        # Shape: [batch, n_prototypes]
        similarities = torch.matmul(query_norm, prototypes_norm.T)
        
        # Apply temperature and softmax
        similarities = similarities / self.temperature
        prototype_weights = F.softmax(similarities, dim=1)
        
        # Blend weight profiles
        # Shape: [batch, n_towers]
        tower_weights = torch.matmul(prototype_weights, self.weight_profiles)
        
        # Softmax to ensure weights sum to 1
        tower_weights = F.softmax(tower_weights, dim=1)
        
        if squeeze_output:
            tower_weights = tower_weights.squeeze(0)
        
        return tower_weights
    
    def get_prototype_similarities(self, query_emb):
        """Get similarity to each prototype (for interpretability)"""
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        
        query_norm = F.normalize(query_emb, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        similarities = torch.matmul(query_norm, prototypes_norm.T)
        return similarities.squeeze()


def apply_tower_weights(embeddings, tower_weights, tower_dims):
    """
    Apply tower weights to embeddings.
    
    Args:
        embeddings: [batch, embedding_dim] or [embedding_dim]
        tower_weights: [n_towers] or [batch, n_towers]
        tower_dims: dict mapping tower names to (start, end) indices
    
    Returns:
        weighted_embeddings: same shape as input
    """
    # Handle single embedding
    single = embeddings.dim() == 1
    if single:
        embeddings = embeddings.unsqueeze(0)
    
    # Handle single weight vector
    if tower_weights.dim() == 1:
        tower_weights = tower_weights.unsqueeze(0)
    
    weighted = torch.zeros_like(embeddings)
    
    tower_list = ['procedures', 'diagnoses', 'demographics', 'place', 'cost', 'pin']
    for i, tower_name in enumerate(tower_list):
        start, end = tower_dims[tower_name]
        weighted[:, start:end] = embeddings[:, start:end] * tower_weights[:, i:i+1]
    
    if single:
        weighted = weighted.squeeze(0)
    
    return weighted


def weighted_cosine_similarity(query_emb, candidate_embs, tower_weights, tower_dims):
    """
    Compute weighted cosine similarity.
    
    Args:
        query_emb: [embedding_dim]
        candidate_embs: [n_candidates, embedding_dim]
        tower_weights: [n_towers]
        tower_dims: dict
    
    Returns:
        similarities: [n_candidates]
    """
    # Apply weights
    weighted_query = apply_tower_weights(query_emb, tower_weights, tower_dims)
    weighted_candidates = apply_tower_weights(candidate_embs, tower_weights, tower_dims)
    
    # Normalize
    weighted_query_norm = F.normalize(weighted_query.unsqueeze(0), p=2, dim=1)
    weighted_candidates_norm = F.normalize(weighted_candidates, p=2, dim=1)
    
    # Cosine similarity
    similarities = torch.matmul(weighted_query_norm, weighted_candidates_norm.T).squeeze(0)
    
    return similarities


# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*80)
print("TRAINING SETUP")
print("="*80)

# Initialize model
model = PrototypeWeightModel(n_prototypes, embedding_dim).to(device)

# Initialize prototypes with mean of each specialty
print("\nInitializing prototypes with specialty means...")
for label, idx in label_to_idx.items():
    specialty_pins = [pin for pin, lbl in pin_to_label.items() if lbl == label]
    specialty_indices = [pin_to_idx[pin] for pin in specialty_pins]
    specialty_embeddings = embeddings[specialty_indices]
    
    mean_embedding = specialty_embeddings.mean(axis=0)
    model.prototypes.data[idx] = torch.FloatTensor(mean_embedding)
    
    print(f"  {label:25s}: {len(specialty_pins):3d} providers")

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
TEMPERATURE = 0.1

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\nHyperparameters:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Temperature: {TEMPERATURE}")

# Convert to tensors
embeddings_tensor = torch.FloatTensor(embeddings).to(device)
tower_dims_tensor = {k: v for k, v in tower_dims.items()}

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("TRAINING")
print("="*80)

# Create training batches (triplets: anchor, positive, negative)
def create_triplet_batches(labeled_pins, pin_to_label, pin_to_idx, batch_size):
    """Create triplet batches for contrastive learning"""
    
    # Group by label
    label_to_pins = defaultdict(list)
    for pin, label in pin_to_label.items():
        label_to_pins[label].append(pin)
    
    batches = []
    labels_list = list(label_to_pins.keys())
    
    # Generate batches
    n_batches = len(labeled_pins) // batch_size
    
    for _ in range(n_batches):
        anchor_pins = []
        positive_pins = []
        negative_pins = []
        
        for _ in range(batch_size):
            # Sample anchor
            anchor_label = random.choice(labels_list)
            anchor_pin = random.choice(label_to_pins[anchor_label])
            
            # Sample positive (same label, different provider)
            positive_candidates = [p for p in label_to_pins[anchor_label] if p != anchor_pin]
            if len(positive_candidates) == 0:
                positive_pin = anchor_pin
            else:
                positive_pin = random.choice(positive_candidates)
            
            # Sample negative (different label)
            negative_label = random.choice([l for l in labels_list if l != anchor_label])
            negative_pin = random.choice(label_to_pins[negative_label])
            
            anchor_pins.append(pin_to_idx[anchor_pin])
            positive_pins.append(pin_to_idx[positive_pin])
            negative_pins.append(pin_to_idx[negative_pin])
        
        batches.append((anchor_pins, positive_pins, negative_pins))
    
    return batches


def triplet_loss(anchor_emb, positive_emb, negative_emb, tower_weights, tower_dims, margin=0.5):
    """Contrastive triplet loss with tower weighting"""
    
    # Compute weighted similarities
    pos_sim = weighted_cosine_similarity(anchor_emb, positive_emb.unsqueeze(0), tower_weights, tower_dims).squeeze()
    neg_sim = weighted_cosine_similarity(anchor_emb, negative_emb.unsqueeze(0), tower_weights, tower_dims).squeeze()
    
    # Triplet loss
    loss = F.relu(margin - pos_sim + neg_sim)
    
    return loss


# Training history
history = {
    'train_loss': [],
    'pos_sim': [],
    'neg_sim': []
}

for epoch in range(EPOCHS):
    model.train()
    
    # Create fresh batches each epoch
    batches = create_triplet_batches(labeled_pins, pin_to_label, pin_to_idx, BATCH_SIZE)
    
    epoch_losses = []
    epoch_pos_sims = []
    epoch_neg_sims = []
    
    for anchor_indices, positive_indices, negative_indices in batches:
        # Get embeddings
        anchor_embs = embeddings_tensor[anchor_indices]
        positive_embs = embeddings_tensor[positive_indices]
        negative_embs = embeddings_tensor[negative_indices]
        
        # Forward pass: predict tower weights for anchors
        tower_weights_batch = model(anchor_embs)
        
        # Compute triplet loss for each sample in batch
        batch_losses = []
        batch_pos_sims = []
        batch_neg_sims = []
        
        for i in range(len(anchor_indices)):
            anchor = anchor_embs[i]
            positive = positive_embs[i]
            negative = negative_embs[i]
            weights = tower_weights_batch[i]
            
            loss = triplet_loss(anchor, positive, negative, weights, tower_dims_tensor)
            
            # Track similarities
            pos_sim = weighted_cosine_similarity(anchor, positive.unsqueeze(0), weights, tower_dims_tensor).item()
            neg_sim = weighted_cosine_similarity(anchor, negative.unsqueeze(0), weights, tower_dims_tensor).item()
            
            batch_losses.append(loss)
            batch_pos_sims.append(pos_sim)
            batch_neg_sims.append(neg_sim)
        
        # Average loss over batch
        loss = torch.stack(batch_losses).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss.item())
        epoch_pos_sims.extend(batch_pos_sims)
        epoch_neg_sims.extend(batch_neg_sims)
    
    scheduler.step()
    
    # Record history
    avg_loss = np.mean(epoch_losses)
    avg_pos_sim = np.mean(epoch_pos_sims)
    avg_neg_sim = np.mean(epoch_neg_sims)
    
    history['train_loss'].append(avg_loss)
    history['pos_sim'].append(avg_pos_sim)
    history['neg_sim'].append(avg_neg_sim)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}: "
              f"Loss={avg_loss:.4f}, "
              f"Pos_sim={avg_pos_sim:.4f}, "
              f"Neg_sim={avg_neg_sim:.4f}, "
              f"Margin={avg_pos_sim - avg_neg_sim:.4f}")

print("\n✓ Training complete!")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

# Save model state
torch.save({
    'model_state_dict': model.state_dict(),
    'n_prototypes': n_prototypes,
    'embedding_dim': embedding_dim,
    'label_to_idx': label_to_idx,
    'idx_to_label': idx_to_label,
    'tower_dims': tower_dims,
    'history': history
}, 'trained_prototype_model.pth')

print("✓ Saved: trained_prototype_model.pth")

# Save metadata
with open('prototype_model_metadata.pkl', 'wb') as f:
    pickle.dump({
        'unique_labels': unique_labels,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'tower_dims': tower_dims,
        'n_prototypes': n_prototypes,
        'embedding_dim': embedding_dim
    }, f)

print("✓ Saved: prototype_model_metadata.pkl")

# ============================================================================
# INSPECT LEARNED WEIGHT PROFILES
# ============================================================================

print("\n" + "="*80)
print("LEARNED WEIGHT PROFILES")
print("="*80)

model.eval()

print(f"\n{'Specialty':<25s} {'Proc':>6s} {'Diag':>6s} {'Demo':>6s} {'Plc':>6s} {'Cost':>6s} {'PIN':>6s}")
print("-" * 80)

weight_profiles = model.weight_profiles.data.cpu().numpy()

for idx in range(n_prototypes):
    label = idx_to_label[idx]
    weights = weight_profiles[idx]
    weights_norm = weights / weights.sum()  # Normalize to see proportions
    
    print(f"{label:<25s} {weights_norm[0]:>6.3f} {weights_norm[1]:>6.3f} "
          f"{weights_norm[2]:>6.3f} {weights_norm[3]:>6.3f} "
          f"{weights_norm[4]:>6.3f} {weights_norm[5]:>6.3f}")

# ============================================================================
# VALIDATION EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("VALIDATION EXAMPLE")
print("="*80)

# Pick a random cancer hospital
cancer_pins = [pin for pin, label in pin_to_label.items() if label == 'Cancer']
query_pin = random.choice(cancer_pins)
query_idx = pin_to_idx[query_pin]
query_emb = embeddings_tensor[query_idx]

print(f"\nQuery provider: PIN {query_pin}")
print(f"True label: Cancer")

# Predict weights
with torch.no_grad():
    predicted_weights = model(query_emb)
    prototype_sims = model.get_prototype_similarities(query_emb)

print(f"\nPredicted tower weights:")
tower_names = ['Procedures', 'Diagnoses', 'Demographics', 'Place', 'Cost', 'PIN']
for i, name in enumerate(tower_names):
    print(f"  {name:15s}: {predicted_weights[i].item():.4f}")

print(f"\nPrototype similarities (top 5):")
top_5_proto_indices = torch.argsort(prototype_sims, descending=True)[:5]
for proto_idx in top_5_proto_indices:
    proto_label = idx_to_label[proto_idx.item()]
    sim = prototype_sims[proto_idx].item()
    print(f"  {proto_label:25s}: {sim:.4f}")

# Find top 10 similar providers
print(f"\nFinding top 10 similar providers...")

with torch.no_grad():
    # Compute weighted similarities to all labeled providers
    labeled_embs = embeddings_tensor[labeled_indices]
    similarities = weighted_cosine_similarity(
        query_emb, labeled_embs, predicted_weights, tower_dims_tensor
    )
    
    # Exclude self
    self_pos = labeled_pins.index(query_pin)
    similarities[self_pos] = -1
    
    # Top 10
    top_10_indices = torch.argsort(similarities, descending=True)[:10]
    
    print(f"\nTop 10 similar providers:")
    print(f"{'Rank':<6s} {'PIN':<12s} {'Label':<25s} {'Similarity':>12s}")
    print("-" * 60)
    
    for rank, idx in enumerate(top_10_indices, 1):
        similar_pin = labeled_pins[idx.item()]
        similar_label = pin_to_label[similar_pin]
        sim = similarities[idx].item()
        
        marker = "✓" if similar_label == 'Cancer' else " "
        print(f"{rank:<6d} {similar_pin:<12s} {similar_label:<25s} {sim:>12.4f} {marker}")

# Count how many are cancer
top_10_labels = [pin_to_label[labeled_pins[idx.item()]] for idx in top_10_indices]
n_cancer = sum(1 for label in top_10_labels if label == 'Cancer')
print(f"\n✓ {n_cancer}/10 are Cancer hospitals")

print("\n" + "="*80)
print("NOTEBOOK 1 COMPLETE")
print("="*80)
print("\nModel saved and validated!")
print("Ready for inference notebooks.")
