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

from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AdamW
from simulation import set_random_seed
from modules.preprocess_kcbert import preprocess
from modules.train import batch_predict
#%%
import gc  

gc.collect()
torch.cuda.empty_cache()  # gpu 캐시를 비워주는 코드
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')

    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='review', help='Dataset options: review')
    
    parser.add_argument('--num_epochs', type=int, default=50, help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', type=int, default=0.00005, help = 'learning_rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
config = {
    'batch_size':32,
    'dataset': 'review'
}
      
#%%
def main():
    config = vars(get_args(debug=False))
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    """dataset & preprocess"""
    df_train, df_test, _, _, test_dataloader = preprocess(config, threshold=79)
    #vocab_size, _, _, test_X_tensor, df_test = preprocess(config)
    
    """model load"""
    model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=2).to(device)
    
    model.load_state_dict(torch.load('./assets/KcBERT.pth'))
    
    predictions = batch_predict(model, test_dataloader, device)
    
    df_test['label'] = predictions
    
    df_test[['id', 'label']].to_csv('submission_KcBert.csv', index=False)
#%%
if __name__ == '__main__':
    main()
    