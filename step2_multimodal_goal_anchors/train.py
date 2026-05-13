import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
import sys

# Import from Step 1
sys.path.append('step1_coordinate_invariant_baseline')
from dataset import get_dataloaders

# Local imports
from model import MultimodalAnchorModel, load_anchors

# Configuration
DATA_DIR = 'data/open/train/'
LABEL_PATH = 'data/open/train_labels.csv'
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'outputs/step2/'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_multimodal_model.pth')
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    anchors = load_anchors().to(DEVICE) # [n_anchors, 3]
    train_loader, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=BATCH_SIZE)
    
    model = MultimodalAnchorModel(n_anchors=len(anchors)).to(DEVICE)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for hist, target, origin in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            hist, target = hist.to(DEVICE), target.to(DEVICE) # target is displacement [B, 3]
            
            optimizer.zero_grad()
            logits, offsets = model(hist) # logits: [B, 6], offsets: [B, 6, 3]
            
            # 1. Find the "Best Anchor" for each sample in batch
            # target: [B, 3], anchors: [6, 3]
            # dist: [B, 6]
            dist = torch.norm(target.unsqueeze(1) - anchors.unsqueeze(0), dim=2)
            best_anchor_idx = torch.argmin(dist, dim=1) # [B]
            
            # 2. Classification Loss (Predicting the best anchor)
            loss_cls = cls_criterion(logits, best_anchor_idx)
            
            # 3. Regression Loss (Predicting offset relative to the chosen anchor)
            # true_offset = target - anchors[best_anchor_idx]
            true_offset = target - anchors[best_anchor_idx]
            
            # Get the predicted offset corresponding to the best anchor
            pred_offset = offsets[torch.arange(target.size(0)), best_anchor_idx]
            loss_reg = reg_criterion(pred_offset, true_offset)
            
            loss = loss_cls + 10.0 * loss_reg # Scaling factor for regression
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation (simplified)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hist, target, origin in val_loader:
                hist, target = hist.to(DEVICE), target.to(DEVICE)
                logits, offsets = model(hist)
                
                dist = torch.norm(target.unsqueeze(1) - anchors.unsqueeze(0), dim=2)
                best_anchor_idx = torch.argmin(dist, dim=1)
                
                loss_cls = cls_criterion(logits, best_anchor_idx)
                true_offset = target - anchors[best_anchor_idx]
                pred_offset = offsets[torch.arange(target.size(0)), best_anchor_idx]
                loss_reg = reg_criterion(pred_offset, true_offset)
                
                val_loss += (loss_cls + 10.0 * loss_reg).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        train()
    else:
        print("Data not found.")
