import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import MosquitoDataset
from model import EqMotion

import argparse

def train(epochs=50, batch_size=64, lr=1e-3):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    train_dir = os.path.join(project_root, "data/open/train")
    labels_path = os.path.join(project_root, "data/open/train_labels.csv")
    
    if not os.path.exists(train_dir):
        train_dir = "data/open/train"
        labels_path = "data/open/train_labels.csv"
        
    save_dir = os.path.join(project_root, "outputs/step4/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Dataset & Dataloader
    full_dataset = MosquitoDataset(train_dir, labels_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = EqMotion().to(device)
    criterion = nn.SmoothL1Loss() # Huber Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for coords, target in pbar:
            coords, target = coords.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for coords, target in val_loader:
                coords, target = coords.to(device), target.to(device)
                output = model(coords)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
