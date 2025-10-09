import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from collections import defaultdict

procedure_df = pd.read_parquet('procedure_df.parquet')
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
member_df = pd.read_parquet('member_df.parquet')
semantic_df_procedure = pd.read_parquet('semantic_df_procedure.parquet')
semantic_df_diagnosis = pd.read_parquet('semantic_df_diagnosis.parquet')
label_df = pd.read_parquet('label_df.parquet')

all_pins = sorted(set(procedure_df['PIN'].unique()) | set(diagnosis_df['PIN'].unique()) | set(member_df['PIN'].unique()))
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

# Tower 1: Procedure embeddings
procedure_docs = defaultdict(list)
for _, row in procedure_df.iterrows():
    code_repeated = [str(row['code'])] * int(row['claims'])
    procedure_docs[row['PIN']].extend(code_repeated)

procedure_documents = [' '.join(procedure_docs.get(pin, ['EMPTY'])) for pin in all_pins]

tfidf_proc = TfidfVectorizer(max_features=5000, min_df=2)
procedure_tfidf = tfidf_proc.fit_transform(procedure_documents)

svd_proc = TruncatedSVD(n_components=128, random_state=42)
tower1_embeddings = svd_proc.fit_transform(procedure_tfidf)
tower1_embeddings = normalize(tower1_embeddings, norm='l2')

print(f"Tower 1: {tower1_embeddings.shape}, variance: {svd_proc.explained_variance_ratio_.sum():.3f}")

# Tower 2: Diagnosis embeddings
diagnosis_docs = defaultdict(list)
for _, row in diagnosis_df.iterrows():
    code_repeated = [str(row['code'])] * int(row['claims'])
    diagnosis_docs[row['PIN']].extend(code_repeated)

diagnosis_documents = [' '.join(diagnosis_docs.get(pin, ['EMPTY'])) for pin in all_pins]

tfidf_diag = TfidfVectorizer(max_features=5000, min_df=2)
diagnosis_tfidf = tfidf_diag.fit_transform(diagnosis_documents)

svd_diag = TruncatedSVD(n_components=128, random_state=42)
tower2_embeddings = svd_diag.fit_transform(diagnosis_tfidf)
tower2_embeddings = normalize(tower2_embeddings, norm='l2')

print(f"Tower 2: {tower2_embeddings.shape}, variance: {svd_diag.explained_variance_ratio_.sum():.3f}")

# Tower 3: Semantic procedure embeddings
semantic_proc_docs = defaultdict(list)
for _, row in semantic_df_procedure.iterrows():
    desc_repeated = [str(row['code_desc'])] * int(row['claims'])
    semantic_proc_docs[row['PIN']].extend(desc_repeated)

semantic_proc_documents = [' '.join(semantic_proc_docs.get(pin, ['EMPTY'])) for pin in all_pins]

tfidf_sem_proc = TfidfVectorizer(max_features=3000, min_df=2, stop_words='english')
semantic_proc_tfidf = tfidf_sem_proc.fit_transform(semantic_proc_documents)

svd_sem_proc = TruncatedSVD(n_components=128, random_state=42)
tower3_embeddings = svd_sem_proc.fit_transform(semantic_proc_tfidf)
tower3_embeddings = normalize(tower3_embeddings, norm='l2')

print(f"Tower 3: {tower3_embeddings.shape}, variance: {svd_sem_proc.explained_variance_ratio_.sum():.3f}")

# Tower 4: Semantic diagnosis embeddings
semantic_diag_docs = defaultdict(list)
for _, row in semantic_df_diagnosis.iterrows():
    desc_repeated = [str(row['code_desc'])] * int(row['claims'])
    semantic_diag_docs[row['PIN']].extend(desc_repeated)

semantic_diag_documents = [' '.join(semantic_diag_docs.get(pin, ['EMPTY'])) for pin in all_pins]

tfidf_sem_diag = TfidfVectorizer(max_features=3000, min_df=2, stop_words='english')
semantic_diag_tfidf = tfidf_sem_diag.fit_transform(semantic_diag_documents)

svd_sem_diag = TruncatedSVD(n_components=128, random_state=42)
tower4_embeddings = svd_sem_diag.fit_transform(semantic_diag_tfidf)
tower4_embeddings = normalize(tower4_embeddings, norm='l2')

print(f"Tower 4: {tower4_embeddings.shape}, variance: {svd_sem_diag.explained_variance_ratio_.sum():.3f}")

# Towers 5-8: Member features
member_indexed = member_df.set_index('PIN')

tower5_features = np.zeros((len(all_pins), 3))
tower6_features = np.zeros((len(all_pins), 2))
tower7_features = np.zeros((len(all_pins), 3))

cost_cols = [col for col in member_df.columns if col.startswith('med_cost_ctg_cd_')]
tower8_features = np.zeros((len(all_pins), len(cost_cols)))

for i, pin in enumerate(all_pins):
    if pin in member_indexed.index:
        row = member_indexed.loc[pin]
        tower5_features[i] = [row['kids_pct'], row['adult_pct'], row['senior_pct']]
        tower6_features[i] = [row['female_pct'], row['male_pct']]
        tower7_features[i] = [row['plc_srv_I_pct'], row['plc_srv_E_pct'], row['plc_srv_O_pct']]
        tower8_features[i] = row[cost_cols].values

tower5_features = normalize(tower5_features, norm='l2')
tower6_features = normalize(tower6_features, norm='l2')
tower7_features = normalize(tower7_features, norm='l2')
tower8_features = normalize(tower8_features, norm='l2')

print(f"Tower 5: {tower5_features.shape}")
print(f"Tower 6: {tower6_features.shape}")
print(f"Tower 7: {tower7_features.shape}")
print(f"Tower 8: {tower8_features.shape}")

# Tower 9: Procedure completeness
procedure_sets = {}
for pin in all_pins:
    procedure_sets[pin] = set(procedure_docs[pin]) if pin in procedure_docs else set()

n_providers = len(all_pins)
tower9_features = np.zeros((n_providers, 20))

for i, pin in enumerate(all_pins):
    pin_procs = procedure_sets[pin]
    if len(pin_procs) == 0:
        continue
    
    similarities = []
    for j, other_pin in enumerate(all_pins):
        if i == j:
            continue
        other_procs = procedure_sets[other_pin]
        if len(other_procs) == 0:
            continue
        
        intersection = len(pin_procs & other_procs)
        union = len(pin_procs | other_procs)
        if union > 0:
            similarities.append(intersection / union)
    
    if similarities:
        similarities_sorted = sorted(similarities, reverse=True)
        percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
        
        for idx, p in enumerate(percentiles):
            perc_idx = int(len(similarities_sorted) * p / 100)
            if perc_idx >= len(similarities_sorted):
                perc_idx = len(similarities_sorted) - 1
            tower9_features[i, idx] = similarities_sorted[perc_idx]
        
        tower9_features[i, 13] = np.mean(similarities)
        tower9_features[i, 14] = np.std(similarities)
        tower9_features[i, 15] = np.max(similarities)
        tower9_features[i, 16] = np.min(similarities)
        tower9_features[i, 17] = len(pin_procs)
        tower9_features[i, 18] = np.median(similarities)
        tower9_features[i, 19] = len(similarities)
    
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{n_providers} providers for Tower 9")

tower9_features = normalize(tower9_features, norm='l2')
print(f"Tower 9: {tower9_features.shape}")

# Create embedding dataframe
pin_name_map = label_df.set_index('PIN')['PIN_name'].to_dict()

embedding_data = {'PIN': all_pins, 'PIN_name': [pin_name_map.get(pin, None) for pin in all_pins]}

for i in range(128):
    embedding_data[f'tower1_proc_emb_{i}'] = tower1_embeddings[:, i]
    embedding_data[f'tower2_diag_emb_{i}'] = tower2_embeddings[:, i]
    embedding_data[f'tower3_sem_proc_emb_{i}'] = tower3_embeddings[:, i]
    embedding_data[f'tower4_sem_diag_emb_{i}'] = tower4_embeddings[:, i]

for i in range(3):
    embedding_data[f'tower5_age_feat_{i}'] = tower5_features[:, i]

for i in range(2):
    embedding_data[f'tower6_gender_feat_{i}'] = tower6_features[:, i]

for i in range(3):
    embedding_data[f'tower7_pos_feat_{i}'] = tower7_features[:, i]

for i in range(tower8_features.shape[1]):
    embedding_data[f'tower8_cost_feat_{i}'] = tower8_features[:, i]

for i in range(20):
    embedding_data[f'tower9_completeness_feat_{i}'] = tower9_features[:, i]

embedding_df = pd.DataFrame(embedding_data)

print(f"\nFinal: {embedding_df.shape}")
embedding_df.to_parquet('provider_embeddings_all_towers.parquet', index=False)
print("Saved")