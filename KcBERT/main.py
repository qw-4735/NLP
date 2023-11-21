import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AdamW
from simulation import set_random_seed
from modules.preprocess_kcbert import preprocess
#from modules.model import KcBert
from modules.train import train, evaluate

#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')

    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='review', help='Dataset options: review')
    #parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension of the model')
    #parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension of the model')
    #parser.add_argument('--output_dim', type=int, default=2, help='output dimension of the model')
    
    parser.add_argument('--num_epochs', type=int, default=30, help='maximum iteration')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', type=int, default=0.001, help = 'learning_rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
# The `config` dictionary is used to store the configuration parameters for the model training. It
# includes the following parameters:

config = {
    'seed':1,
    'lr' : 0.000001,
    'num_epochs' : 50,
    'batch_size':32,
    'dataset': 'review'
}

def main():
    config = vars(get_args(debug=False))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    """dataset & preprocess"""
    _, _, train_dataloader, valid_dataloader, _ = preprocess(config, threshold=79)
        
    #next(iter(train_dataloader))['input_ids'].shape
    
    """model"""
    model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2).to(device)
    #model
    #model.bert.embeddings
    #model.bert.encoder.layer
    #model.classifier
    
    #model.state_dict().keys()
    #total_params = sum(p.numel() for p in model.parameters())  # 모델 파라미터 수
    
    """Pretrained BERT 레이어를 freezing"""
    for name, param in model.bert.named_parameters():
        if 'encoder.layer' in name:
            layer_num = int(name.split(".")[2])
            if layer_num < 8:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # freezing layer 확인하는 코드
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Freezing Layer: {name}")
                  
    # bert_layer 전체 freezing 코드              
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    
    """분류 레이어만 학습"""    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])  # requires_grad가 True인 파라미터만 업데이트
    
    """train"""
    set_random_seed(config['seed'])

    train_losses = []
    train_accuracys = []
    train_f1s = []
    
    val_losses = []
    val_accuracys = []
    val_f1s = []
    for epoch in range(config['num_epochs']):
        
        train_loss, train_accuracy, train_f1 = train(model, criterion, train_dataloader, optimizer, device)
        val_loss, val_accuracy, val_f1 = evaluate(model, criterion, valid_dataloader, device)
        
        train_losses.append(round(train_loss,3))
        train_accuracys.append(train_accuracy)
        train_f1s.append(train_f1)
        val_losses.append(round(val_loss,3))
        val_accuracys.append(val_accuracy)
        val_f1s.append(val_f1)
        
        print(f'epoch : {epoch}, train_loss: {train_loss: .5f}, train_accuracy : {train_accuracy}, train_f1 : {train_f1 : .5f}, val_loss: {val_loss: .5f}, val_accuracy: {val_accuracy}, val_f1 : {val_f1 : .5f}')

    """model save"""
    torch.save(model.state_dict(), './assets/KcBERT.pth')
    

#%%
if __name__ == '__main__':
    main()

plt.plot(train_losses, color='b', label='train')
plt.plot(val_losses, color='r', label='valid' )
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('training curve(loss)')
plt.legend()
plt.show()


plt.plot(train_accuracys, color='b', label='train')
plt.plot(val_accuracys, color='r', label='valid' )
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('training curve(accuracy)')
plt.legend()
plt.show()

plt.plot(train_f1s, color='b', label='train')
plt.plot(val_f1s, color='r', label='valid' )
plt.xlabel('epochs')
plt.ylabel('f1')
plt.title('training curve(f1)')
plt.legend()
plt.show()