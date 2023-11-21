#%%
import os

print('JAVA_HOME' in os.environ)

os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-21\bin\server'
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
#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#os.chdir(r'C:\Users\11 home\Documents\review dataset\dataset')
#%%
from modules.preprocess_func import tokenize, create_word_list, remove_rare_word, texts_to_sequences, pad_sequences
#%%
def preprocess_tokenize(config):
    
    if config['dataset'] == 'review':
        df_train = pd.read_csv('.//data//review dataset//dataset//train.csv')
        df_test = pd.read_csv('.//data//review dataset//dataset//test.csv')
        
        train_str_column = df_train.select_dtypes(include='object').columns.item()
        test_str_column = df_test.select_dtypes(include='object').columns.item()
        
        # 1) 영어, 한글을 제외한 불필요한 특수문자 제거
        df_train[train_str_column] = df_train[train_str_column].map(lambda x : re.sub('[^A-Za-z ㄱ-ㅎ ㅏ-ㅣ가-힣]', '', x)) # 영어, 한글만 포함
        df_test[test_str_column] = df_test[test_str_column].map(lambda x : re.sub('[^A-Za-z ㄱ-ㅎ ㅏ-ㅣ 가-힣]', '', x))

        # 2),3) 토큰화 및 불용어 제거
        X_train = tokenize(df_train)
        X_test = tokenize(df_test)

        # 4) 학습데이터, 검증데이터 나누기
        y_train = np.array(df_train['label'])

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    else:
        raise ValueError('Not supported dataset!')
    
    return df_train, df_test, X_train, X_valid, X_test, y_train, y_valid

# 5) 단어 집합 만들기
def preprocess_rare_word(X_train):
    
    word_list = create_word_list(X_train)
    word_counts = Counter(word_list)
    
    vocab = sorted(word_counts, key=word_counts.get, reverse=True) 
    
    total_cnt = len(word_counts)  # 단어의 수
    rare_cnt = remove_rare_word(word_counts)
    
    # 전체 단어 개수 중 빈도 수 2이하인 단어는 제거.
    vocab_size = total_cnt - rare_cnt
    vocab = vocab[:vocab_size]

    word_to_index = {}
    word_to_index['<PAD>'] = 0
    word_to_index['<UNK>'] = 1

    for index, word in enumerate(vocab):
        word_to_index[word] = index + 2
    
    return vocab_size, word_to_index

# 6) 정수 인코딩        
def preprocess_encoding(X_train, X_valid, X_test, word_to_index):
    
    encoded_X_train = texts_to_sequences(X_train, word_to_index)
    encoded_X_valid = texts_to_sequences(X_valid, word_to_index)
    encoded_X_test = texts_to_sequences(X_test, word_to_index)
    
    return encoded_X_train, encoded_X_valid, encoded_X_test

# 7) 패딩
def preprocess_padding(encoded_X_train, encoded_X_valid, encoded_X_test):    
    
    padded_X_train = pad_sequences(encoded_X_train)
    padded_X_valid = pad_sequences(encoded_X_valid)
    padded_X_test = pad_sequences(encoded_X_test)
    
    return padded_X_train, padded_X_valid, padded_X_test

# 학습/검증/테스트 데이터 파이토치 텐서로 변환 -> 데이터로더로 변환
def preprocess_dataset(padded_X_train, padded_X_valid, padded_X_test, y_train, y_valid, config):    
    
    train_X_tensor = torch.LongTensor(padded_X_train)
    valid_X_tensor = torch.LongTensor(padded_X_valid)
    test_X_tensor = torch.LongTensor(padded_X_test)
    
    train_label_tensor = torch.LongTensor(np.array(y_train))
    valid_label_tensor = torch.LongTensor(np.array(y_valid))

    train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_label_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'])

    valid_dataset = torch.utils.data.TensorDataset(valid_X_tensor, valid_label_tensor)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size= config['batch_size'])

    return train_loader, valid_loader, test_X_tensor
    
    #return vocab_size, train_loader, valid_loader, test_X_tensor, df_test

#%%
































