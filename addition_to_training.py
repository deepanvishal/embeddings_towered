# Add to Phase 1 metadata (line ~590):
metadata = {
    'latent_dim': PHASE1_LATENT_DIM,
    'specialty_code_indices': all_specialty_codes,  # ← ADD THIS (which columns to filter)
    # ... rest unchanged
}

# Add to final metadata (line ~1155):
final_metadata = {
    'specialty_code_indices': all_specialty_codes,  # ← ADD THIS
    # ... rest unchanged
}
