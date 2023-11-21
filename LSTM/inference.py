#%%
import os

print('JAVA_HOME' in os.environ)

os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-21\bin\server'
#%%
import os
os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import re
import tqdm
import random
from collections import Counter
from konlpy.tag import Okt

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.preprocess import preprocess_tokenize, preprocess_rare_word, preprocess_encoding, preprocess_padding, preprocess_dataset
from modules.model import LSTM
#%%
import argparse

def get_args(debug):
    parser = argparse.ArgumentParser(description='parameters')

    parser.add_argument('--seed', type=int, default=1, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='review', help='Dataset options: review')
    parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension of the model')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension of the model')
    parser.add_argument('--output_dim', type=int, default=2, help='output dimension of the model')
    
    parser.add_argument('--num_epochs', type=int, default=30, help='maximum iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', type=int, default=0.001, help = 'learning_rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
#%%
config = {
    'seed':1,
    'embedding_dim' : 100,
    'hidden_dim' : 128,
    'output_dim' : 2,
    'lr' : 0.001,
    'num_epochs' : 30,
    'batch_size':64,
    'dataset': 'review'
}
#%%

def main():
    config = vars(get_args(debug=False))
    device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config['cuda'] = torch.cuda.is_available()
    
    """dataset & preprocess"""
    df_train, df_test, X_train, X_valid, X_test, y_train, y_valid = preprocess_tokenize(config)
    vocab_size, word_to_index = preprocess_rare_word(X_train)
    encoded_X_train, encoded_X_valid, encoded_X_test = preprocess_encoding(X_train, X_valid, X_test, word_to_index)
    padded_X_train, padded_X_valid, padded_X_test = preprocess_padding(encoded_X_train, encoded_X_valid, encoded_X_test)
    _, _, test_X_tensor = preprocess_dataset(padded_X_train, padded_X_valid, padded_X_test, y_train, y_valid, config)
    
    #vocab_size, _, _, test_X_tensor, df_test = preprocess(config)
    
    """model load"""
    model = LSTM(vocab_size, config).to(device)
    
    model.load_state_dict(torch.load('./assets/LSTM.pth'))
    pred = model(test_X_tensor)
    
    result = torch.argmax(pred, dim=1)
    
    df_test['label'] = result
    
    df_test[['id', 'label']].to_csv('submission_lstm.csv', index=False)
#%%
if __name__ == '__main__':
    main()
    
    