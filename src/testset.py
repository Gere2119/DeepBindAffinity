import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import ast
from tqdm import tqdm

# ==================== Configuration ====================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_PATH = "D:/Paper/PLBA/data"  # Update with your new dataset path
MODEL_DIR = "D:/Paper/PLBA/src/model"  # Where your 10 fold models are saved
MAX_SEQ_LEN = 1024
MAX_SMI_LEN = 256
BATCH_SIZE = 32

# ==================== Dataset Class ====================
class TestDataset(Dataset):
    def __init__(self, data_path, max_seq_len=1024, max_smi_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        
        # Load test data - adjust file names as needed
        self.df = pd.read_csv(Path(data_path) / 'seq_data_core2013.csv')
        self.affinity = pd.read_csv(Path(data_path) / 'affinity.csv', index_col=0)['affinity'].to_dict()
        
        # Character dictionaries
        self.smi_char = {'<MASK>': 0,'C': 1, ')': 2, '(': 3, 'c': 4, 'O': 5, ']': 6, '[': 7,
                        '@': 8, '1': 9, '=': 10, 'H': 11, 'N': 12, '2': 13, 'n': 14,
                        '3': 15, 'o': 16, '+': 17, '-': 18, 'S': 19, 'F': 20, 'p': 21,
                        'l': 22, '/': 23, '4': 24, '#': 25, 'B': 26, '\\': 27, '5': 28,
                        'r': 29, 's': 30, '6': 31, 'I': 32, '7': 33, '%': 34, '8': 35,
                        'e': 36, 'P': 37, '9': 38, 'R': 39, 'u': 40, '0': 41, 'i': 42,
                        '.': 43, 'A': 44, 't': 45, 'h': 46, 'V': 47, 'g': 48, 'b': 49,
                        'Z': 50, 'T': 51, 'M': 52}
        
        self.protein_char = {'<MASK>': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4,
                           'F': 5, 'G': 6, 'H': 7, 'K': 8,
                           'I': 9, 'L': 10, 'M': 11, 'N': 12,
                           'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                           'T': 17, 'V': 18, 'Y': 19, 'W': 20}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['PDBname']
        
        # Process SMILES
        smi = row['Smile']
        smi_encoded = self._encode_smiles(smi)
        
        # Process protein sequence
        seq = row['Sequence']
        seq_encoded = self._encode_sequence(seq)
        
        # Process pocket positions
        positions = ast.literal_eval(row['Position'])
        pocket_encoded = self._encode_pocket(seq, positions)
        
        # Get affinity if available
        affinity = torch.tensor(
            [self.affinity.get(pdb_id, 0)],  # 0 if no affinity available
            dtype=torch.float32,
            device=device
        )
        
        return pdb_id, smi_encoded, seq_encoded, pocket_encoded, affinity

    def _encode_smiles(self, smi):
        label = np.zeros(self.max_smi_len)
        for i, ch in enumerate(smi[:self.max_smi_len]):
            label[i] = self.smi_char.get(ch, 0)  # Default to 0 (<MASK>)
        return torch.tensor(label, device=device).long()

    def _encode_sequence(self, seq):
        label = np.zeros(self.max_seq_len)
        for i, aa in enumerate(seq[:self.max_seq_len]):
            label[i] = self.protein_char.get(aa, 0)  # Default to 0 (<MASK>)
        return torch.tensor(label, device=device).long()

    def _encode_pocket(self, seq, positions):
        masked_seq = ['<MASK>'] * len(seq)
        for pos in positions:
            if pos <= len(seq):
                masked_seq[pos-1] = seq[pos-1]
        return self._encode_sequence(''.join(masked_seq))

# ==================== Model Architecture ====================
class InteractionPredictor(nn.Module):
    def __init__(self, smi_vocab_size=53, seq_vocab_size=21, embed_dim=128):
        super().__init__()
        # SMILES embedding and encoder
        self.smi_embed = nn.Embedding(smi_vocab_size, embed_dim, padding_idx=0)
        self.smi_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # Protein sequence embedding and encoder
        self.seq_embed = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.seq_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # Pocket embedding and encoder
        self.pocket_embed = nn.Embedding(seq_vocab_size, embed_dim, padding_idx=0)
        self.pocket_encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, smi, seq, pocket):
        # SMILES processing
        smi_emb = self.smi_embed(smi)
        smi_out, _ = self.smi_encoder(smi_emb)
        smi_feat = smi_out.mean(dim=1)
        
        # Sequence processing
        seq_emb = self.seq_embed(seq)
        seq_out, _ = self.seq_encoder(seq_emb)
        seq_feat = seq_out.mean(dim=1)
        
        # Pocket processing
        pocket_emb = self.pocket_embed(pocket)
        pocket_out, _ = self.pocket_encoder(pocket_emb)
        pocket_feat = pocket_out.mean(dim=1)
        
        # Combine and predict
        combined = torch.cat([smi_feat, seq_feat, pocket_feat], dim=1)
        return self.head(combined)

# ==================== Testing Function ====================
def test_models(test_loader, model_paths):
    results = {
        'pdb_id': [],
        'true_affinity': [],
        'predicted_affinity': [],
        'model_fold': []
    }
    
    metrics = {
        'mse': [],
        'mae': []
    }
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    for fold, model_path in enumerate(model_paths):
        print(f"\nTesting model from fold {fold + 1}")
        
        # Load model
        model = InteractionPredictor().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        fold_mse = 0
        fold_mae = 0
        
        with torch.no_grad():
            for pdb_ids, smi, seq, pocket, affinity in tqdm(test_loader, desc=f"Fold {fold + 1}"):
                preds = model(smi, seq, pocket).squeeze()
                
                # Store results
                results['pdb_id'].extend(pdb_ids)
                results['true_affinity'].extend(affinity.cpu().numpy())
                results['predicted_affinity'].extend(preds.cpu().numpy())
                results['model_fold'].extend([fold+1] * len(pdb_ids))
                
                # Calculate metrics
                fold_mse += criterion_mse(preds, affinity.squeeze()).item()
                fold_mae += criterion_mae(preds, affinity.squeeze()).item()
        
        # Average metrics for this fold
        fold_mse /= len(test_loader)
        fold_mae /= len(test_loader)
        
        metrics['mse'].append(fold_mse)
        metrics['mae'].append(fold_mae)
        
        print(f"Fold {fold + 1} - MSE: {fold_mse:.4f}, MAE: {fold_mae:.4f}")
    
    return results, metrics

# ==================== Main Execution ====================
if __name__ == "__main__":
    # Initialize test dataset
    test_dataset = TestDataset(DATA_PATH, MAX_SEQ_LEN, MAX_SMI_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get paths to all 10 trained models
    model_paths = [f"{MODEL_DIR}/Fold {i+1} best_model.ckpt" for i in range(10)]
    
    # Verify all model files exist
    for path in model_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
    
    print(f"Found {len(model_paths)} pretrained models")
    print(f"Testing on {len(test_dataset)} samples")
    
    # Test all models
    results, metrics = test_models(test_loader, model_paths)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate ensemble predictions (mean across all models)
    ensemble_df = results_df.groupby('pdb_id').agg({
        'true_affinity': 'first',
        'predicted_affinity': 'mean'
    }).reset_index()
    
    # Calculate ensemble metrics
    ensemble_mse = np.mean(metrics['mse'])
    ensemble_mae = np.mean(metrics['mae'])
    
    # Save results
    results_df.to_csv("individual_model_predictions.csv", index=False)
    ensemble_df.to_csv("ensemble_predictions.csv", index=False)
    
    # Print final metrics
    print("\n=== Final Metrics ===")
    print(f"Average MSE across all folds: {ensemble_mse:.4f}")
    print(f"Average MAE across all folds: {ensemble_mae:.4f}")
    
    # Print per-fold metrics
    print("\n=== Per-Fold Metrics ===")
    for fold, (mse, mae) in enumerate(zip(metrics['mse'], metrics['mae'])):
        print(f"Fold {fold + 1}: MSE = {mse:.4f}, MAE = {mae:.4f}")