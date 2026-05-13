import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import BaselineGRU
import os
from tqdm.auto import tqdm

# Configuration
DATA_DIR = 'data/open/train/'
LABEL_PATH = 'data/open/train_labels.csv'
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = 'outputs/step1/'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_baseline_model.pth')
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    train_loader, val_loader = get_dataloaders(DATA_DIR, LABEL_PATH, batch_size=BATCH_SIZE)
    
    model = BaselineGRU().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for hist, target, origin in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            hist, target = hist.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(hist)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hist, target, origin in val_loader:
                hist, target = hist.to(DEVICE), target.to(DEVICE)
                output = model(hist)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Saved best model with Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    if os.path.exists(DATA_DIR):
        train()
    else:
        print(f"Data directory not found at {DATA_DIR}. Please check the path.")
