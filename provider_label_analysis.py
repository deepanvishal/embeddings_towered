import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import gc

def get_label_centroid(label):
    mask = embedding_df['true_label'] == label
    embeddings = embedding_df.loc[mask, [f'emb_{i}' for i in range(64)]].values
    return embeddings.mean(axis=0)

def get_provider_embedding(pin):
    mask = embedding_df['PIN'] == pin
    return embedding_df.loc[mask, [f'emb_{i}' for i in range(64)]].values[0]

def calculate_similarity(provider_pin, label):
    provider_emb = get_provider_embedding(provider_pin)
    label_centroid = get_label_centroid(label)
    similarity = cosine_similarity([provider_emb], [label_centroid])[0][0]
    return similarity

def get_provider_codes(pin, code_type, top_n=10):
    mask = (amt_smry_df['PIN'] == pin) & (amt_smry_df['code_type'] == code_type)
    codes = amt_smry_df.loc[mask, ['code', 'code_desc', 'claims']]
    
    if len(codes) == 0:
        return pd.DataFrame()
    
    grouped = codes.groupby(['code', 'code_desc'], as_index=False)['claims'].sum()
    total = grouped['claims'].sum()
    grouped['%'] = (grouped['claims'] / total * 100).round(2)
    result = grouped.nlargest(top_n, 'claims')
    
    if code_type == 'Procedure':
        result = result.rename(columns={'code': 'Procedure_Code', 'code_desc': 'Procedure_Description', 'claims': 'Claims'})
    else:
        result = result.rename(columns={'code': 'ICD10_Code', 'code_desc': 'ICD10_Description', 'claims': 'Claims'})
    
    return result[result.columns[:4]]

def get_label_codes(label, code_type, top_n=10):
    label_pins = embedding_df[embedding_df['true_label'] == label]['PIN'].values
    mask = (amt_smry_df['PIN'].isin(label_pins)) & (amt_smry_df['code_type'] == code_type)
    codes = amt_smry_df.loc[mask, ['code', 'code_desc', 'claims']]
    
    if len(codes) == 0:
        return pd.DataFrame()
    
    grouped = codes.groupby(['code', 'code_desc'], as_index=False)['claims'].sum()
    total = grouped['claims'].sum()
    grouped['%'] = (grouped['claims'] / total * 100).round(2)
    result = grouped.nlargest(top_n, 'claims')
    
    if code_type == 'Procedure':
        result = result.rename(columns={'code': 'Procedure_Code', 'code_desc': 'Procedure_Description', 'claims': 'Claims'})
    else:
        result = result.rename(columns={'code': 'ICD10_Code', 'code_desc': 'ICD10_Description', 'claims': 'Claims'})
    
    return result[result.columns[:4]]

def get_common_codes(pin, label, code_type, top_n=10):
    provider_codes = get_provider_codes(pin, code_type, 50)
    label_codes = get_label_codes(label, code_type, 50)
    
    if len(provider_codes) == 0 or len(label_codes) == 0:
        return pd.DataFrame()
    
    code_col = 'Procedure_Code' if code_type == 'Procedure' else 'ICD10_Code'
    desc_col = 'Procedure_Description' if code_type == 'Procedure' else 'ICD10_Description'
    
    common = provider_codes.merge(label_codes, on=[code_col, desc_col], suffixes=('_Provider', '_Label'))
    common['Total_Claims'] = common['Claims_Provider'] + common['Claims_Label']
    
    result = common.nlargest(top_n, 'Total_Claims')
    return result[[code_col, desc_col, 'Claims_Provider', 'Claims_Label', '%_Provider', '%_Label']]

def get_demographics(pin):
    demo_cols = ['peds_pct', 'adults_pct', 'seniors_pct', 'Female_pct', 'Inpatient_pct', 'Emergency_pct']
    mask = new_member_df['PIN'] == pin
    if not mask.any():
        return pd.Series(index=demo_cols, dtype=float)
    return new_member_df.loc[mask, demo_cols].iloc[0]

def get_label_demographics(label):
    demo_cols = ['peds_pct', 'adults_pct', 'seniors_pct', 'Female_pct', 'Inpatient_pct', 'Emergency_pct']
    label_pins = embedding_df[embedding_df['true_label'] == label]['PIN'].values
    mask = new_member_df['PIN'].isin(label_pins)
    if not mask.any():
        return pd.Series(index=demo_cols, dtype=float)
    return new_member_df.loc[mask, demo_cols].mean()

def get_costs(pin):
    cost_cols = [f'med_cost_ctg_cd_{str(i).zfill(3)}_pct' for i in [1,2,3,4,5,6,7,8,9,10,16]]
    cost_labels = ['IP Facility', 'AMB Facility', 'Emergency', 'Specialty Physician', 'PCP Physician', 
                   'Radiology', 'LAB', 'Home Health', 'Mental Health', 'Medical Rx', 'Other']
    mask = new_member_df['PIN'] == pin
    if not mask.any():
        return pd.Series(index=cost_labels, dtype=float)
    costs = new_member_df.loc[mask, cost_cols].iloc[0]
    costs.index = cost_labels
    return costs

def get_label_costs(label):
    cost_cols = [f'med_cost_ctg_cd_{str(i).zfill(3)}_pct' for i in [1,2,3,4,5,6,7,8,9,10,16]]
    cost_labels = ['IP Facility', 'AMB Facility', 'Emergency', 'Specialty Physician', 'PCP Physician', 
                   'Radiology', 'LAB', 'Home Health', 'Mental Health', 'Medical Rx', 'Other']
    label_pins = embedding_df[embedding_df['true_label'] == label]['PIN'].values
    mask = new_member_df['PIN'].isin(label_pins)
    if not mask.any():
        return pd.Series(index=cost_labels, dtype=float)
    costs = new_member_df.loc[mask, cost_cols].mean()
    costs.index = cost_labels
    return costs

def create_pca_plot(provider_pin, label_a, label_b, provider_name):
    plt.figure(figsize=(14, 8))
    
    emb_cols = [f'emb_{i}' for i in range(64)]
    
    mask_a = embedding_df['true_label'] == label_a
    mask_b = embedding_df['true_label'] == label_b
    
    embeddings_a = embedding_df.loc[mask_a, emb_cols].values
    embeddings_b = embedding_df.loc[mask_b, emb_cols].values
    
    combined = np.vstack([embeddings_a, embeddings_b])
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(combined)
    
    count_a = len(embeddings_a)
    pca_a = pca_coords[:count_a]
    pca_b = pca_coords[count_a:]
    
    centroid_a_emb = embeddings_a.mean(axis=0)
    centroid_b_emb = embeddings_b.mean(axis=0)
    
    centroids_pca = pca.transform([centroid_a_emb, centroid_b_emb])
    centroid_a_pca = centroids_pca[0]
    centroid_b_pca = centroids_pca[1]
    
    provider_emb = get_provider_embedding(provider_pin)
    provider_pca = pca.transform([provider_emb])[0]
    
    plt.scatter(pca_a[:, 0], pca_a[:, 1], alpha=0.5, label=label_a, s=30, color='lightblue')
    plt.scatter(pca_b[:, 0], pca_b[:, 1], alpha=0.5, label=label_b, s=30, color='lightcoral')
    
    plt.scatter(centroid_a_pca[0], centroid_a_pca[1], marker='X', s=400, 
                color='blue', edgecolors='black', linewidth=2, label=f'{label_a} Centroid')
    plt.scatter(centroid_b_pca[0], centroid_b_pca[1], marker='X', s=400, 
                color='red', edgecolors='black', linewidth=2, label=f'{label_b} Centroid')
    
    plt.scatter(provider_pca[0], provider_pca[1], marker='*', s=500, 
                color='gold', edgecolors='black', linewidth=2, label=f'{provider_name}')
    
    dist_a = np.linalg.norm(provider_emb - centroid_a_emb)
    dist_b = np.linalg.norm(provider_emb - centroid_b_emb)
    
    plt.plot([provider_pca[0], centroid_a_pca[0]], [provider_pca[1], centroid_a_pca[1]], 
             'b--', alpha=0.7, linewidth=2, label=f'Distance to {label_a}: {dist_a:.3f}')
    plt.plot([provider_pca[0], centroid_b_pca[0]], [provider_pca[1], centroid_b_pca[1]], 
             'r--', alpha=0.7, linewidth=2, label=f'Distance to {label_b}: {dist_b:.3f}')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'PCA: {provider_name} vs {label_a} vs {label_b}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    gc.collect()

def run_analysis(provider_pin, label_a, label_b):
    provider_name = new_scores_with_names[new_scores_with_names['PIN'] == provider_pin]['PIN_name'].iloc[0]
    
    print("="*80)
    print(f"PROVIDER LABEL COMPARISON ANALYSIS")
    print("="*80)
    print(f"Provider: {provider_name} (PIN: {provider_pin})")
    print(f"Label A: {label_a}")
    print(f"Label B: {label_b}")
    print()
    
    sim_a = calculate_similarity(provider_pin, label_a)
    sim_b = calculate_similarity(provider_pin, label_b)
    
    print("="*80)
    print("1. SIMILARITY SCORES")
    print("="*80)
    print(f"Similarity to {label_a}: {sim_a:.4f}")
    print(f"Similarity to {label_b}: {sim_b:.4f}")
    print()
    
    print("="*80)
    print("2. PROVIDER TOP 10 PROCEDURES")
    print("="*80)
    provider_proc = get_provider_codes(provider_pin, 'Procedure', 10)
    display(provider_proc)
    print()
    
    print("="*80)
    print(f"3. TOP PROCEDURES - {label_a}")
    print("="*80)
    label_a_proc = get_label_codes(label_a, 'Procedure', 10)
    display(label_a_proc)
    print()
    
    print("="*80)
    print(f"4. COMMON PROCEDURES WITH {label_a}")
    print("="*80)
    common_a_proc = get_common_codes(provider_pin, label_a, 'Procedure', 10)
    if len(common_a_proc) > 0:
        display(common_a_proc)
    else:
        print("No common procedures found")
    print()
    
    print("="*80)
    print(f"5. TOP PROCEDURES - {label_b}")
    print("="*80)
    label_b_proc = get_label_codes(label_b, 'Procedure', 10)
    display(label_b_proc)
    print()
    
    print("="*80)
    print(f"6. COMMON PROCEDURES WITH {label_b}")
    print("="*80)
    common_b_proc = get_common_codes(provider_pin, label_b, 'Procedure', 10)
    if len(common_b_proc) > 0:
        display(common_b_proc)
    else:
        print("No common procedures found")
    print()
    
    print("="*80)
    print("7. PROVIDER TOP 10 DIAGNOSES")
    print("="*80)
    provider_diag = get_provider_codes(provider_pin, 'Diagnosis', 10)
    display(provider_diag)
    print()
    
    print("="*80)
    print(f"8. TOP DIAGNOSES - {label_a}")
    print("="*80)
    label_a_diag = get_label_codes(label_a, 'Diagnosis', 10)
    display(label_a_diag)
    print()
    
    print("="*80)
    print(f"9. COMMON DIAGNOSES WITH {label_a}")
    print("="*80)
    common_a_diag = get_common_codes(provider_pin, label_a, 'Diagnosis', 10)
    if len(common_a_diag) > 0:
        display(common_a_diag)
    else:
        print("No common diagnoses found")
    print()
    
    print("="*80)
    print(f"10. TOP DIAGNOSES - {label_b}")
    print("="*80)
    label_b_diag = get_label_codes(label_b, 'Diagnosis', 10)
    display(label_b_diag)
    print()
    
    print("="*80)
    print(f"11. COMMON DIAGNOSES WITH {label_b}")
    print("="*80)
    common_b_diag = get_common_codes(provider_pin, label_b, 'Diagnosis', 10)
    if len(common_b_diag) > 0:
        display(common_b_diag)
    else:
        print("No common diagnoses found")
    print()
    
    print("="*80)
    print("12. DEMOGRAPHICS COMPARISON")
    print("="*80)
    provider_demo = get_demographics(provider_pin)
    label_a_demo = get_label_demographics(label_a)
    label_b_demo = get_label_demographics(label_b)
    
    demo_comparison = pd.DataFrame({
        'Metric': provider_demo.index,
        provider_name: provider_demo.values,
        f'{label_a} Avg': label_a_demo.values,
        f'{label_b} Avg': label_b_demo.values
    })
    display(demo_comparison)
    print()
    
    print("="*80)
    print("13. COST DISTRIBUTION COMPARISON")
    print("="*80)
    provider_cost = get_costs(provider_pin)
    label_a_cost = get_label_costs(label_a)
    label_b_cost = get_label_costs(label_b)
    
    cost_comparison = pd.DataFrame({
        'Cost Category': provider_cost.index,
        provider_name: provider_cost.values,
        f'{label_a} Avg': label_a_cost.values,
        f'{label_b} Avg': label_b_cost.values
    })
    display(cost_comparison)
    print()
    
    print("="*80)
    print("14. PCA VISUALIZATION")
    print("="*80)
    create_pca_plot(provider_pin, label_a, label_b, provider_name)

labels = sorted(embedding_df[embedding_df['true_label'] != 'Unlabeled']['true_label'].unique())

provider_options = []
for _, row in new_scores_with_names.iterrows():
    if row['PIN'] in embedding_df['PIN'].values:
        provider_options.append((f"{row['PIN_name']} ({row['PIN']})", row['PIN']))

provider_dropdown = widgets.Dropdown(
    options=provider_options,
    description='Provider:',
    style={'description_width': '100px'},
    layout={'width': '600px'}
)

label_a_dropdown = widgets.Dropdown(
    options=labels,
    description='Label A:',
    style={'description_width': '100px'},
    layout={'width': '400px'}
)

label_b_dropdown = widgets.Dropdown(
    options=labels,
    description='Label B:',
    style={'description_width': '100px'},
    layout={'width': '400px'}
)

run_button = widgets.Button(
    description='Run Analysis',
    button_style='primary',
    layout={'width': '200px'}
)

output = widgets.Output()

def on_run_click(b):
    with output:
        clear_output(wait=True)
        provider_pin = provider_dropdown.value
        label_a = label_a_dropdown.value
        label_b = label_b_dropdown.value
        run_analysis(provider_pin, label_a, label_b)

run_button.on_click(on_run_click)

display(widgets.VBox([
    widgets.HTML("<h2>Provider Label Comparison Analysis</h2>"),
    provider_dropdown,
    label_a_dropdown,
    label_b_dropdown,
    run_button,
    output
]))
