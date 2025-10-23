import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Filter to only the 7,990 labeled providers
labeled_pins = list(pin_to_label.keys())

print(f"Total providers in data: {len(pin_list)}")
print(f"Labeled providers: {len(labeled_pins)}")

# Find indices of labeled pins in pin_list (alignment already guaranteed)
pin_to_idx = {pin: idx for idx, pin in enumerate(pin_list)}
labeled_indices = [pin_to_idx[pin] for pin in labeled_pins if pin in pin_to_idx]

print(f"Matched indices: {len(labeled_indices)}")

# Filter embedding_df to only labeled providers
embedding_df_filtered = embedding_df[embedding_df['PIN'].isin(labeled_pins)].reset_index(drop=True)

# Filter matrices to only labeled providers (rows) - alignment preserved
diag_matrix_filtered = diag_matrix[labeled_indices, :]
proc_matrix_filtered = proc_matrix[labeled_indices, :]

# Create filtered pin_list - alignment preserved
filtered_pin_list = [pin_list[i] for i in labeled_indices]

print(f"\nFiltered shapes:")
print(f"embedding_df: {embedding_df_filtered.shape}")
print(f"diag_matrix: {diag_matrix_filtered.shape}")
print(f"proc_matrix: {proc_matrix_filtered.shape}")
print(f"filtered_pin_list length: {len(filtered_pin_list)}")

# Verify we have all labeled providers
print(f"\nAll labeled PINs found: {len(labeled_indices) == len(labeled_pins)}")
