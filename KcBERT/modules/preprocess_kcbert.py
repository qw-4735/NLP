#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from modules.preprocess_kcbert_func import find_max_length_with_threshold
#%%
class CustomDataset_train(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
    
#%%
class CustomDataset_test(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
#%%

def preprocess(config, threshold=79):
    
    if config['dataset'] == 'review':
        df_train = pd.read_csv('.//data//train.csv')
        df_test = pd.read_csv('.//data//test.csv')
        
        #train_str_column = df_train.select_dtypes(include='object').columns.item()
        #test_str_column = df_test.select_dtypes(include='object').columns.item()
        
        """remove stopwords"""
        #df_train[train_str_column] = df_train[train_str_column].map(lambda x : re.sub('[^A-Za-z ㄱ-ㅎ ㅏ-ㅣ가-힣]', '', x)) # 영어, 한글만 포함
        #df_test[test_str_column] = df_test[test_str_column].map(lambda x : re.sub('[^A-Za-z ㄱ-ㅎ ㅏ-ㅣ 가-힣]', '', x))

        """train set"""
        train_texts = df_train['document'].to_list()
        labels = df_train['label'].to_list()
        
        """test set"""
        test_texts = df_test['document'].to_list()
        
        tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        
        max_length = find_max_length_with_threshold(threshold, tokenizer, train_texts) # threshold 기준으로 선정한 max_length
        #max_length = max(len(review) for review in tokenizer(train_texts)['input_ids'])  # 실제 문장들 중 max_length
            
        labels = torch.tensor(labels, dtype=torch.long)
        train_dataset = CustomDataset_train(train_texts, labels, tokenizer, max_length)
        test_dataset = CustomDataset_test(test_texts, tokenizer, max_length)
        #train_dataset[0]['input_ids'].shape
        
        train, valid = train_test_split(train_dataset, test_size=0.2, random_state=0)
        
        train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True)
        valid_dataloader = DataLoader(valid, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    else:
        raise ValueError('Not supported dataset!')
    
    return df_train, df_test, train_dataloader, valid_dataloader, test_dataloader
#%%
# a = train_texts[0]       
#tokenizer(a, padding = 'max_length', truncation = True, max_length = 19, return_tensors = 'pt')
#tokenizer('영상이나')
