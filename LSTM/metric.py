import torch

def calculate_metrics(pred, labels):
    
    tp = (pred * labels).sum().item()
    fp = (pred * (1-labels)).sum().item()
    tn = ((1-pred)*labels).sum().item()
    fn = ((1-pred)*labels).sum().item()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return precision, recall, f1_score