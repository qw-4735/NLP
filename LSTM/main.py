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
from modules.train import train, evaluate
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

def main():
    config = vars(get_args(debug=False))
    config['cuda'] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    """dataset & preprocess"""
    df_train, df_test, X_train, X_valid, X_test, y_train, y_valid = preprocess_tokenize(config)
    vocab_size, word_to_index = preprocess_rare_word(X_train)
    encoded_X_train, encoded_X_valid, encoded_X_test = preprocess_encoding(X_train, X_valid, X_test, word_to_index)
    padded_X_train, padded_X_valid, padded_X_test = preprocess_padding(encoded_X_train, encoded_X_valid, encoded_X_test)
    train_loader, valid_loader, test_X_tensor = preprocess_dataset(padded_X_train, padded_X_valid, padded_X_test, y_train, y_valid, config)
    
    #vocab_size, train_loader, valid_loader, _, _ = preprocess(config)
    
    """model"""
    model = LSTM(vocab_size, config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    """train"""
    random.seed(1)

    train_losses = []
    train_accuracys = []
    train_f1s = []
    
    val_losses = []
    val_accuracys = []
    val_f1s = []
    for epoch in range(config['num_epochs']):
        
        train_loss, train_accuracy, train_f1 = train(model, criterion, train_loader, optimizer, device)
        val_loss, val_accuracy, val_f1 = evaluate(model, criterion, valid_loader, device)
        
        train_losses.append(round(train_loss,3))
        train_accuracys.append(train_accuracy)
        train_f1s.append(train_f1)
        val_losses.append(round(val_loss,3))
        val_accuracys.append(val_accuracy)
        val_f1s.append(val_f1)
        
        print(f'epoch : {epoch}, train_loss: {train_loss: .5f}, train_accuracy : {train_accuracy}, train_f1 : {train_f1 : .5f}, val_loss: {val_loss: .5f}, val_accuracy: {val_accuracy}, val_f1 : {val_f1 : .5f}')

    """model save"""
    torch.save(model.state_dict(), './assets/LSTM.pth')
    

#%%
if __name__ == '__main__':
    main()


# plt.plot(train_losses, color='b', label='train')
# plt.plot(val_losses, color='r', label='valid' )
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('training curve(loss)')
# plt.legend()
# plt.show()


# plt.plot(train_accuracys, color='b', label='train')
# plt.plot(val_accuracys, color='r', label='valid' )
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.title('training curve(accuracy)')
# plt.legend()
# plt.show()

# plt.plot(train_f1s, color='b', label='train')
# plt.plot(val_f1s, color='r', label='valid' )
# plt.xlabel('epochs')
# plt.ylabel('f1')
# plt.title('training curve(f1)')
# plt.legend()
# plt.show()


    