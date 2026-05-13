import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import ArenaAwareAnchorModel, load_anchors
import os
from tqdm.auto import tqdm

# Configuration
DATA_DIR = 'data/open/train/'
LABEL_PATH = 'data/open/train_labels.csv'
BATCH_SIZE = 128
EPOCHS = 70  # Increased epochs for more complex learning
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'outputs/step3/'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_step3_model.pth')
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    anchors = load_anchors().to(DEVICE)
    train_loader, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=BATCH_SIZE)
    
    model = ArenaAwareAnchorModel(n_anchors=len(anchors)).to(DEVICE)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss() # Using Huber loss for more stability
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for hist, arena, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            hist, arena, target = hist.to(DEVICE), arena.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            logits, offsets = model(hist, arena)
            
            # Find best anchor
            dist = torch.norm(target.unsqueeze(1) - anchors.unsqueeze(0), dim=2)
            best_anchor_idx = torch.argmin(dist, dim=1)
            
            # Multi-task Loss
            loss_cls = cls_criterion(logits, best_anchor_idx)
            true_offset = target - anchors[best_anchor_idx]
            pred_offset = offsets[torch.arange(target.size(0)), best_anchor_idx]
            loss_reg = reg_criterion(pred_offset, true_offset)
            
            loss = loss_cls + 20.0 * loss_reg # Slightly higher weight on regression
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hist, arena, target in val_loader:
                hist, arena, target = hist.to(DEVICE), arena.to(DEVICE), target.to(DEVICE)
                logits, offsets = model(hist, arena)
                
                dist = torch.norm(target.unsqueeze(1) - anchors.unsqueeze(0), dim=2)
                best_anchor_idx = torch.argmin(dist, dim=1)
                
                loss_cls = cls_criterion(logits, best_anchor_idx)
                true_offset = target - anchors[best_anchor_idx]
                pred_offset = offsets[torch.arange(target.size(0)), best_anchor_idx]
                loss_reg = reg_criterion(pred_offset, true_offset)
                
                val_loss += (loss_cls + 20.0 * loss_reg).item()
        
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
        print("Data directory not found.")
