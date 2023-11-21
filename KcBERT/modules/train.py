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
    for step, batch in enumerate(train_loader):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(input_ids, attention_mask)
        y_pred = y_pred.logits
        
        loss = criterion(y_pred, labels)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        
        """accuracy"""
        pred_idx = torch.argmax(y_pred, dim=1)
        correct = (pred_idx == labels).sum().item()
        train_correct += correct
        
        """fl-score"""
        precision, recall, f1_score = calculate_metrics(pred_idx, labels)
        train_f1 += f1_score
        
        train_total += labels.size(0)
    
    avg_f1 = train_f1 / len(train_loader)    
    avg_accuracy = train_correct / train_total 
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, avg_accuracy, avg_f1


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    
    model.eval()
    
    val_loss, val_correct, val_total, val_f1 = 0, 0, 0, 0
    
    for step, val_batch in enumerate(loader):
        
        val_input_ids = val_batch['input_ids'].to(device)
        val_attention_mask = val_batch['attention_mask'].to(device)
        val_labels = val_batch['label'].to(device)


        y_pred = model(val_input_ids, val_attention_mask)
        y_pred = y_pred.logits
        
        loss = criterion(y_pred, val_labels)
        val_loss += loss.item()
        
        """accuracy"""
        pred_idx = torch.argmax(y_pred, dim=1)
        correct = (pred_idx == val_labels).sum().item()
        val_correct += correct
        
        
        """fl-score"""
        precision, recall, f1_score = calculate_metrics(pred_idx, val_labels)
        val_f1 += f1_score
        
        val_total += val_labels.size(0)
    
    avg_f1 = val_f1 / len(loader)
    avg_loss = val_loss / len(loader)
    avg_accuracy = val_correct / val_total
    
    return avg_loss, avg_accuracy, avg_f1

#%%
def batch_predict(model, test_loader, device):
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits

            # 확률값으로 변환
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # 각 배치에 대한 예측 결과 저장
            _, predicted_labels = torch.max(probs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
    
    return predictions


#%%
from modules.preprocess_kcbert import preprocess

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config = {
    'seed':1,
    'lr' : 0.000001,
    'num_epochs' : 50,
    'batch_size':32,
    'dataset': 'review'
}
_, _, train_dataloader, valid_dataloader, _ = preprocess(config, threshold=79)

input_ids = next(iter(train_dataloader))['input_ids'].to(device)
input_ids.shape
input_ids[0]

attention_mask = next(iter(train_dataloader))['attention_mask'].to(device) 
attention_mask.shape
attention_mask[0]

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2).to(device)

y_pred = model(input_ids, attention_mask)
y_pred.logits.shape

# input_ids = next(iter(test_dataloader))['input_ids'].to(device)
# input_ids.shape
# attention_mask = next(iter(test_dataloader))['attention_mask'].to(device) 
# attention_mask.shape
# outputs.logits.shape    
# predictions1 = batch_predict(model, test_dataloader, device)            
# len(predictions1)          
            