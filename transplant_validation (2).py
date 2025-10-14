import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import gc

def get_providers_for_procedure(procedure_code):
    mask = (amt_smry_df['code'] == procedure_code) & (amt_smry_df['code_type'] == 'Procedure')
    provider_claims = amt_smry_df.loc[mask, ['PIN', 'claims']].groupby('PIN', as_index=False)['claims'].sum()
    provider_claims = provider_claims.sort_values('claims', ascending=False)
    return provider_claims

def get_provider_claims_for_procedure(pin, procedure_code):
    mask = (amt_smry_df['PIN'] == pin) & (amt_smry_df['code'] == procedure_code) & (amt_smry_df['code_type'] == 'Procedure')
    claims = amt_smry_df.loc[mask, 'claims'].sum()
    return claims if claims > 0 else 0

def get_similar_providers(pin, top_n=5):
    emb_cols = [f'emb_{i}' for i in range(64)]
    
    provider_mask = embedding_df['PIN'] == pin
    if not provider_mask.any():
        return []
    
    provider_emb = embedding_df.loc[provider_mask, emb_cols].values[0]
    all_embeddings = embedding_df[emb_cols].values
    
    similarities = cosine_similarity([provider_emb], all_embeddings)[0]
    
    similar_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in similar_indices:
        similar_pin = embedding_df.iloc[idx]['PIN']
        if similar_pin == pin:
            continue
        results.append({
            'PIN': similar_pin,
            'similarity': similarities[idx]
        })
        if len(results) >= top_n:
            break
    
    return results

def create_tsne_plot(procedure_code, procedure_desc):
    emb_cols = [f'emb_{i}' for i in range(64)]
    all_embeddings = embedding_df[emb_cols].values
    all_pins = embedding_df['PIN'].values
    
    performed_pins = set(amt_smry_df[(amt_smry_df['code'] == procedure_code) & 
                                      (amt_smry_df['code_type'] == 'Procedure')]['PIN'].unique())
    
    performed_mask = np.array([pin in performed_pins for pin in all_pins])
    
    print(f"Running t-SNE on {len(all_embeddings)} providers...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(14, 10))
    
    not_performed = tsne_coords[~performed_mask]
    performed = tsne_coords[performed_mask]
    
    plt.scatter(not_performed[:, 0], not_performed[:, 1], 
                alpha=0.5, s=30, color='lightgray', label='Did NOT perform procedure')
    plt.scatter(performed[:, 0], performed[:, 1], 
                alpha=0.7, s=50, color='red', label='Performed procedure')
    
    plt.title(f't-SNE: Providers by {procedure_desc}\n({len(performed_pins)} performed, {len(all_pins) - len(performed_pins)} did not perform)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    gc.collect()

def run_validation_analysis(procedure_code, procedure_desc):
    print("="*80)
    print(f"TRANSPLANT PROCEDURE VALIDATION ANALYSIS")
    print("="*80)
    print(f"Selected Procedure: {procedure_desc}")
    print(f"Procedure Code: {procedure_code}")
    print()
    
    provider_claims = get_providers_for_procedure(procedure_code)
    
    if len(provider_claims) == 0:
        print("No providers found for this procedure.")
        return
    
    top_5_providers = provider_claims.head(5)
    
    print("="*80)
    print(f"TOP 5 PROVIDERS BY CLAIMS")
    print("="*80)
    
    for idx, row in top_5_providers.iterrows():
        pin = row['PIN']
        claims = row['claims']
        provider_name = new_scores_with_names[new_scores_with_names['PIN'] == pin]['PIN_name'].iloc[0] if pin in new_scores_with_names['PIN'].values else f"Provider {pin}"
        print(f"{provider_name} (PIN: {pin}) - {claims} claims")
    print()
    
    for provider_idx, row in top_5_providers.iterrows():
        pin = row['PIN']
        claims = row['claims']
        provider_name = new_scores_with_names[new_scores_with_names['PIN'] == pin]['PIN_name'].iloc[0] if pin in new_scores_with_names['PIN'].values else f"Provider {pin}"
        
        print("="*80)
        print(f"PROVIDER {provider_idx + 1}: {provider_name}")
        print("="*80)
        print(f"Claims for selected procedure: {claims}")
        print()
        print("Top 5 Similar Providers:")
        print("-"*80)
        
        similar_providers = get_similar_providers(pin, top_n=5)
        
        results = []
        for i, similar in enumerate(similar_providers, 1):
            similar_pin = similar['PIN']
            similarity = similar['similarity']
            similar_name = new_scores_with_names[new_scores_with_names['PIN'] == similar_pin]['PIN_name'].iloc[0] if similar_pin in new_scores_with_names['PIN'].values else f"Provider {similar_pin}"
            similar_claims = get_provider_claims_for_procedure(similar_pin, procedure_code)
            
            results.append({
                'Rank': f"Provider{provider_idx + 1} -> {chr(64 + i)}",
                'PIN': similar_pin,
                'PIN_Name': similar_name,
                'Similarity': f"{similarity:.4f}",
                'Claims': similar_claims if similar_claims > 0 else "Not performed"
            })
        
        results_df = pd.DataFrame(results)
        display(results_df)
        print()
    
    print("="*80)
    print("t-SNE VISUALIZATION")
    print("="*80)
    create_tsne_plot(procedure_code, procedure_desc)

transplant_with_claims = []
for _, row in transplant_df.iterrows():
    procedure_code = row['procedure_code']
    procedure_desc = row['procedure_description']
    mask = (amt_smry_df['code'] == procedure_code) & (amt_smry_df['code_type'] == 'Procedure')
    total_claims = amt_smry_df.loc[mask, 'claims'].sum()
    transplant_with_claims.append({
        'procedure_code': procedure_code,
        'procedure_description': procedure_desc,
        'total_claims': total_claims
    })

transplant_with_claims = pd.DataFrame(transplant_with_claims)
transplant_with_claims = transplant_with_claims.sort_values('total_claims', ascending=False)

procedure_options = [(f"{row['procedure_description']} - {row['procedure_code']} (Total Claims: {int(row['total_claims'])})", 
                      (row['procedure_code'], row['procedure_description'])) 
                     for _, row in transplant_with_claims.iterrows()]

procedure_dropdown = widgets.Combobox(
    options=[desc for desc, _ in procedure_options],
    description='Search Procedure:',
    placeholder='Type to search...',
    ensure_option=False,
    style={'description_width': '120px'},
    layout={'width': '800px'}
)

run_button = widgets.Button(
    description='Run Validation Analysis',
    button_style='primary',
    layout={'width': '200px'}
)

output = widgets.Output()

def on_run_click(b):
    with output:
        clear_output(wait=True)
        
        selected_text = procedure_dropdown.value
        
        matching = [code_desc for desc, code_desc in procedure_options if desc == selected_text]
        
        if not matching:
            print("Please select a valid procedure from the list.")
            return
        
        procedure_code, procedure_desc = matching[0]
        run_validation_analysis(procedure_code, procedure_desc)

run_button.on_click(on_run_click)

display(widgets.VBox([
    widgets.HTML("<h2>Transplant Procedure Validation Analysis</h2>"),
    widgets.HTML("<p>Search and select a transplant procedure to validate embedding similarity.</p>"),
    procedure_dropdown,
    run_button,
    output
]))
