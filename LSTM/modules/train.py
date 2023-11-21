#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.classification import BinaryF1Score

from metric import calculate_metrics



#%%
def train(model, criterion, train_loader, optimizer, device):
    model.train()
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_f1 = 0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        
        """accuracy"""
        pred_idx = torch.argmax(y_pred, dim=1)
        correct = (pred_idx == y_batch).sum().item()
        train_correct += correct
        
        """fl-score"""
        precision, recall, f1_score = calculate_metrics(pred_idx, y_batch)
        train_f1 += f1_score
        
        train_total += y_batch.size(0)
    
    avg_f1 = train_f1 / len(train_loader)    
    avg_accuracy = train_correct / train_total 
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, avg_accuracy, avg_f1


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    
    model.eval()
    
    val_loss, val_correct, val_total, val_f1 = 0, 0, 0, 0
    
    for step, (x_batch, y_batch) in enumerate(loader):
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        
        loss = criterion(y_pred, y_batch)
        val_loss += loss.item()
        
        """accuracy"""
        pred_idx = torch.argmax(y_pred, dim=1)
        correct = (pred_idx == y_batch).sum().item()
        val_correct += correct
        
        
        """fl-score"""
        precision, recall, f1_score = calculate_metrics(pred_idx, y_batch)
        val_f1 += f1_score
        
        val_total += y_batch.size(0)
    
    avg_f1 = val_f1 / len(loader)
    avg_loss = val_loss / len(loader)
    avg_accuracy = val_correct / val_total
    
    return avg_loss, avg_accuracy, avg_f1