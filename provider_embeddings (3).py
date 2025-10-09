import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

procedure_df = pd.read_parquet('procedure_df.parquet')
diagnosis_df = pd.read_parquet('diagnosis_df.parquet')
member_df = pd.read_parquet('member_df.parquet')
label_df = pd.read_parquet('label_df.parquet')

all_pins = sorted(set(procedure_df['PIN'].unique()) | set(diagnosis_df['PIN'].unique()))
pin_to_idx = {pin: idx for idx, pin in enumerate(all_pins)}

procedure_codes = sorted(procedure_df['code'].unique())
diagnosis_codes = sorted(diagnosis_df['code'].unique())
proc_to_idx = {code: idx for idx, code in enumerate(procedure_codes)}
diag_to_idx = {code: idx for idx, code in enumerate(diagnosis_codes)}

n_proc = len(procedure_codes)
n_diag = len(diagnosis_codes)

print(f"Providers: {len(all_pins)}, Procedures: {n_proc}, Diagnoses: {n_diag}")

procedure_grouped = procedure_df.groupby('PIN').apply(
    lambda x: dict(zip(x['code'], x['claims']))
).to_dict()

diagnosis_grouped = diagnosis_df.groupby('PIN').apply(
    lambda x: dict(zip(x['code'], x['claims']))
).to_dict()

member_indexed = member_df.set_index('PIN')
label_indexed = label_df.set_index('PIN')

class ProviderDataset(Dataset):
    def __init__(self, pins, proc_grouped, diag_grouped, member_indexed, proc_to_idx, diag_to_idx, n_proc, n_diag):
        self.pins = pins
        self.proc_grouped = proc_grouped
        self.diag_grouped = diag_grouped
        self.member_indexed = member_indexed
        self.proc_to_idx = proc_to_idx
        self.diag_to_idx = diag_to_idx
        self.n_proc = n_proc
        self.n_diag = n_diag
        
        cost_cols = [col for col in member_indexed.columns if col.startswith('med_cost_ctg_cd_')]
        self.cost_cols = cost_cols
        self.n_cost = len(cost_cols)
    
    def __len__(self):
        return len(self.pins)
    
    def __getitem__(self, idx):
        pin = self.pins[idx]
        
        proc_vec = torch.zeros(self.n_proc)
        if pin in self.proc_grouped:
            for code, claims in self.proc_grouped[pin].items():
                if code in self.proc_to_idx:
                    proc_vec[self.proc_to_idx[code]] = np.log1p(claims)
        
        diag_vec = torch.zeros(self.n_diag)
        if pin in self.diag_grouped:
            for code, claims in self.diag_grouped[pin].items():
                if code in self.diag_to_idx:
                    diag_vec[self.diag_to_idx[code]] = np.log1p(claims)
        
        if pin in self.member_indexed.index:
            row = self.member_indexed.loc[pin]
            demo_vec = torch.tensor([
                row['kids_pct'], row['adult_pct'], row['senior_pct'],
                row['female_pct'], row['male_pct'],
                row['plc_srv_I_pct'], row['plc_srv_E_pct'], row['plc_srv_O_pct']
            ], dtype=torch.float32)
            cost_vec = torch.tensor(row[self.cost_cols].values, dtype=torch.float32)
        else:
            demo_vec = torch.zeros(8)
            cost_vec = torch.zeros(self.n_cost)
        
        return proc_vec, diag_vec, demo_vec, cost_vec, pin

dataset = ProviderDataset(all_pins, procedure_grouped, diagnosis_grouped, member_indexed, 
                          proc_to_idx, diag_to_idx, n_proc, n_diag)

class MultiTowerEncoder(nn.Module):
    def __init__(self, n_proc, n_diag, n_demo=8, n_cost=10):
        super().__init__()
        
        self.tower1 = nn.Sequential(
            nn.Linear(n_proc, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )
        
        self.tower2 = nn.Sequential(
            nn.Linear(n_diag, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )
        
        self.tower3 = nn.Sequential(
            nn.Linear(n_demo, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.tower4 = nn.Sequential(
            nn.Linear(n_cost, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, proc, diag, demo, cost):
        tower1_emb = self.tower1(proc)
        tower2_emb = self.tower2(diag)
        tower3_emb = self.tower3(demo)
        tower4_emb = self.tower4(cost)
        
        return tower1_emb, tower2_emb, tower3_emb, tower4_emb

model = MultiTowerEncoder(n_proc, n_diag, n_demo=8, n_cost=dataset.n_cost).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

label_encoder = LabelEncoder()
labeled_pins = label_indexed.index.tolist()
labels_encoded = label_encoder.fit_transform(label_indexed['label'].values)
pin_to_label = dict(zip(labeled_pins, labels_encoded))

class ContrastiveDataset(Dataset):
    def __init__(self, base_dataset, pin_to_label):
        self.base_dataset = base_dataset
        self.pin_to_label = pin_to_label
        
        self.labeled_indices = []
        for idx, pin in enumerate(base_dataset.pins):
            if pin in pin_to_label:
                self.labeled_indices.append(idx)
        
        self.label_to_indices = {}
        for idx in self.labeled_indices:
            pin = base_dataset.pins[idx]
            label = pin_to_label[pin]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.labeled_indices)
    
    def __getitem__(self, idx):
        anchor_idx = self.labeled_indices[idx]
        anchor_pin = self.base_dataset.pins[anchor_idx]
        anchor_label = self.pin_to_label[anchor_pin]
        
        positive_candidates = [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
        if positive_candidates:
            positive_idx = np.random.choice(positive_candidates)
        else:
            positive_idx = anchor_idx
        
        negative_label = np.random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        
        anchor_data = self.base_dataset[anchor_idx]
        positive_data = self.base_dataset[positive_idx]
        negative_data = self.base_dataset[negative_idx]
        
        return anchor_data, positive_data, negative_data

contrastive_dataset = ContrastiveDataset(dataset, pin_to_label)
train_loader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

print(f"Training samples: {len(contrastive_dataset)}")

def multi_tower_triplet_loss(anchor_towers, positive_towers, negative_towers, margin=1.0):
    total_loss = 0
    for i in range(len(anchor_towers)):
        pos_dist = (anchor_towers[i] - positive_towers[i]).pow(2).sum(1)
        neg_dist = (anchor_towers[i] - negative_towers[i]).pow(2).sum(1)
        loss = torch.relu(pos_dist - neg_dist + margin)
        total_loss += loss.mean()
    return total_loss / len(anchor_towers)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (anchor_data, positive_data, negative_data) in enumerate(train_loader):
        anchor_proc, anchor_diag, anchor_demo, anchor_cost, _ = anchor_data
        positive_proc, positive_diag, positive_demo, positive_cost, _ = positive_data
        negative_proc, negative_diag, negative_demo, negative_cost, _ = negative_data
        
        anchor_proc = anchor_proc.to(device)
        anchor_diag = anchor_diag.to(device)
        anchor_demo = anchor_demo.to(device)
        anchor_cost = anchor_cost.to(device)
        
        positive_proc = positive_proc.to(device)
        positive_diag = positive_diag.to(device)
        positive_demo = positive_demo.to(device)
        positive_cost = positive_cost.to(device)
        
        negative_proc = negative_proc.to(device)
        negative_diag = negative_diag.to(device)
        negative_demo = negative_demo.to(device)
        negative_cost = negative_cost.to(device)
        
        optimizer.zero_grad()
        
        anchor_towers = model(anchor_proc, anchor_diag, anchor_demo, anchor_cost)
        positive_towers = model(positive_proc, positive_diag, positive_demo, positive_cost)
        negative_towers = model(negative_proc, negative_diag, negative_demo, negative_cost)
        
        loss = multi_tower_triplet_loss(anchor_towers, positive_towers, negative_towers)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs} completed, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining complete. Generating embeddings...")

model.eval()
tower1_all = []
tower2_all = []
tower3_all = []
tower4_all = []
all_pins_list = []

inference_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

with torch.no_grad():
    for batch in inference_loader:
        proc, diag, demo, cost, pins = batch
        proc = proc.to(device)
        diag = diag.to(device)
        demo = demo.to(device)
        cost = cost.to(device)
        
        t1, t2, t3, t4 = model(proc, diag, demo, cost)
        
        tower1_all.append(t1.cpu().numpy())
        tower2_all.append(t2.cpu().numpy())
        tower3_all.append(t3.cpu().numpy())
        tower4_all.append(t4.cpu().numpy())
        all_pins_list.extend(pins)

tower1_embeddings = np.vstack(tower1_all)
tower2_embeddings = np.vstack(tower2_all)
tower3_embeddings = np.vstack(tower3_all)
tower4_embeddings = np.vstack(tower4_all)

pin_name_map = label_df.set_index('PIN')['PIN_name'].to_dict()

embedding_data = {
    'PIN': all_pins_list,
    'PIN_name': [pin_name_map.get(pin, None) for pin in all_pins_list]
}

for i in range(128):
    embedding_data[f'tower1_proc_emb_{i}'] = tower1_embeddings[:, i]
    embedding_data[f'tower2_diag_emb_{i}'] = tower2_embeddings[:, i]

for i in range(32):
    embedding_data[f'tower3_demo_emb_{i}'] = tower3_embeddings[:, i]
    embedding_data[f'tower4_cost_emb_{i}'] = tower4_embeddings[:, i]

embedding_df = pd.DataFrame(embedding_data)

print(f"\nFinal embeddings: {embedding_df.shape}")
print(f"Tower 1 (Procedures): 128 dims")
print(f"Tower 2 (Diagnoses): 128 dims")
print(f"Tower 3 (Demographics): 32 dims")
print(f"Tower 4 (Cost): 32 dims")

embedding_df.to_parquet('provider_embeddings_multi_tower.parquet', index=False)
torch.save(model.state_dict(), 'multi_tower_encoder.pth')
print("Embeddings and model saved")