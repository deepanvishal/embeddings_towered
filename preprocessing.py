import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")
procedure_df = pd.read_parquet('procedure_df.parquet')
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
member_df = pd.read_parquet('member_df.parquet')
label_df = pd.read_parquet('label_df.parquet')

print(f"procedure_df: {procedure_df.shape}")
print(f"diagnosis_df: {diagnosis_df.shape}")
print(f"member_df: {member_df.shape}")
print(f"label_df: {label_df.shape}")

# ============================================================================
# CREATE UNIFIED PIN LIST
# ============================================================================
print("\nCreating unified PIN list...")
all_pins = sorted(set(procedure_df['PIN'].unique()) | 
                  set(diagnosis_df['PIN'].unique()) | 
                  set(member_df['PIN'].unique()))
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

print(f"Total PINs: {len(all_pins)}")

# ============================================================================
# CREATE AMT_SMRY VECTORS (COMBINED)
# ============================================================================
print("\nBuilding amt_smry vectors (procedure + diagnosis combined)...")
amt_smry_df = pd.concat([procedure_df, diagnosis_df], ignore_index=True)
amt_grouped = amt_smry_df.groupby(['PIN', 'code'])['claims'].sum().reset_index()

amt_codes = sorted(amt_grouped['code'].unique())
amt_to_idx = {code: idx for idx, code in enumerate(amt_codes)}
n_amt = len(amt_codes)

row_indices = amt_grouped['PIN'].map(pin_to_idx).values
col_indices = amt_grouped['code'].map(amt_to_idx).values
values = np.log1p(amt_grouped['claims'].values).astype(np.float32)

amt_matrix = coo_matrix((values, (row_indices, col_indices)), 
                        shape=(len(all_pins), n_amt),
                        dtype=np.float32).tocsr()

print(f"amt_smry matrix: {amt_matrix.shape}, density: {amt_matrix.nnz / (amt_matrix.shape[0] * amt_matrix.shape[1]):.4f}")

# ============================================================================
# CREATE PROCEDURE VECTORS (SEPARATE)
# ============================================================================
print("\nBuilding procedure vectors...")
proc_grouped = procedure_df.groupby(['PIN', 'code'])['claims'].sum().reset_index()

procedure_codes = sorted(proc_grouped['code'].unique())
proc_to_idx = {code: idx for idx, code in enumerate(procedure_codes)}
n_proc = len(procedure_codes)

row_indices = proc_grouped['PIN'].map(pin_to_idx).values
col_indices = proc_grouped['code'].map(proc_to_idx).values
values = np.log1p(proc_grouped['claims'].values).astype(np.float32)

proc_matrix = coo_matrix((values, (row_indices, col_indices)), 
                         shape=(len(all_pins), n_proc),
                         dtype=np.float32).tocsr()

print(f"procedure matrix: {proc_matrix.shape}, density: {proc_matrix.nnz / (proc_matrix.shape[0] * proc_matrix.shape[1]):.4f}")

# ============================================================================
# CREATE DIAGNOSIS VECTORS (SEPARATE)
# ============================================================================
print("\nBuilding diagnosis vectors...")
diag_grouped = diagnosis_df.groupby(['PIN', 'code'])['claims'].sum().reset_index()

diagnosis_codes = sorted(diag_grouped['code'].unique())
diag_to_idx = {code: idx for idx, code in enumerate(diagnosis_codes)}
n_diag = len(diagnosis_codes)

row_indices = diag_grouped['PIN'].map(pin_to_idx).values
col_indices = diag_grouped['code'].map(diag_to_idx).values
values = np.log1p(diag_grouped['claims'].values).astype(np.float32)

diag_matrix = coo_matrix((values, (row_indices, col_indices)), 
                         shape=(len(all_pins), n_diag),
                         dtype=np.float32).tocsr()

print(f"diagnosis matrix: {diag_matrix.shape}, density: {diag_matrix.nnz / (diag_matrix.shape[0] * diag_matrix.shape[1]):.4f}")

# ============================================================================
# CREATE MEMBER VECTORS
# ============================================================================
print("\nBuilding member vectors...")
member_indexed = member_df.set_index('PIN')
member_cols = [col for col in member_df.columns if col != 'PIN']
n_member = len(member_cols)

member_matrix = np.zeros((len(all_pins), n_member), dtype=np.float32)
for i, pin in enumerate(all_pins):
    if pin in member_indexed.index:
        member_matrix[i] = member_indexed.loc[pin].values.astype(np.float32)

print(f"member matrix: {member_matrix.shape}")

# ============================================================================
# PROCESS LABELS
# ============================================================================
print("\nProcessing labels...")
label_encoder = LabelEncoder()
labeled_pins = label_df.iloc[:, 0].tolist()
labels_encoded = label_encoder.fit_transform(label_df.iloc[:, 1].values)
pin_to_label = dict(zip(labeled_pins, labels_encoded))

print(f"Labeled PINs: {len(pin_to_label)}")

# ============================================================================
# SAVE VECTORS
# ============================================================================
print("\nSaving vectors...")
sp.save_npz('amt_smry_vectors.npz', amt_matrix)
sp.save_npz('procedure_vectors.npz', proc_matrix)
sp.save_npz('diagnosis_vectors.npz', diag_matrix)
np.save('member_vectors.npy', member_matrix)
np.save('all_pins.npy', np.array(all_pins))
with open('pin_to_label.pkl', 'wb') as f:
    pickle.dump(pin_to_label, f)

metadata = {
    'n_amt': n_amt,
    'n_proc': n_proc,
    'n_diag': n_diag,
    'n_member': n_member,
    'n_pins': len(all_pins)
}
with open('vectors_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\nSaved:")
print(f"  amt_smry_vectors.npz: {amt_matrix.shape}")
print(f"  procedure_vectors.npz: {proc_matrix.shape}")
print(f"  diagnosis_vectors.npz: {diag_matrix.shape}")
print(f"  member_vectors.npy: {member_matrix.shape}")
print(f"  all_pins.npy: {len(all_pins)}")
print(f"  pin_to_label.pkl: {len(pin_to_label)} labeled PINs")
print(f"  vectors_metadata.pkl")

# ============================================================================
# LOAD VECTORS (IN NEW NOTEBOOK)
# ============================================================================
print("\n" + "="*80)
print("LOADING VECTORS (use this in your next notebook)")
print("="*80)

amt_matrix = sp.load_npz('amt_smry_vectors.npz')
proc_matrix = sp.load_npz('procedure_vectors.npz')
diag_matrix = sp.load_npz('diagnosis_vectors.npz')
member_matrix = np.load('member_vectors.npy')
all_pins = np.load('all_pins.npy', allow_pickle=True).tolist()

with open('pin_to_label.pkl', 'rb') as f:
    pin_to_label = pickle.load(f)

with open('vectors_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"\nLoaded vectors:")
print(f"  amt_smry: {amt_matrix.shape}")
print(f"  procedure: {proc_matrix.shape}")
print(f"  diagnosis: {diag_matrix.shape}")
print(f"  member: {member_matrix.shape}")
print(f"  pins: {len(all_pins)}")
print(f"  pin_to_label: {len(pin_to_label)} labeled PINs")
print(f"\nMetadata: {metadata}")

# ============================================================================
# CONVERT TO TENSORS
# ============================================================================
print("\nConverting to tensors...")

amt_tensor = torch.FloatTensor(amt_matrix.toarray())
proc_tensor = torch.FloatTensor(proc_matrix.toarray())
diag_tensor = torch.FloatTensor(diag_matrix.toarray())
member_tensor = torch.FloatTensor(member_matrix)

amt_tensor = amt_tensor.to(device)
proc_tensor = proc_tensor.to(device)
diag_tensor = diag_tensor.to(device)
member_tensor = member_tensor.to(device)

print(f"\nTensors on {device}:")
print(f"  amt_tensor: {amt_tensor.shape}")
print(f"  proc_tensor: {proc_tensor.shape}")
print(f"  diag_tensor: {diag_tensor.shape}")
print(f"  member_tensor: {member_tensor.shape}")

print("\n" + "="*80)
print("COMPLETE - Ready for model training")
print("="*80)
