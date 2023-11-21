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
    
# 불용어 제거
def remove_stopwords(text):    
    okt = Okt()
    tokens = okt.pos(text)
    result = [word for word, pos in tokens if pos not in ['Josa', 'Eomi', 'Punctuation']]  # 품사가 조사, 어미, 구두점이면 제거함.
    return result

# 토큰화
def tokenize(df):
    X_list = []
    for sentence in df['document']:
        tokenized_sentence = remove_stopwords(sentence)
        X_list.append(tokenized_sentence)
    return X_list

# 단어 집합 만들기
def create_word_list(sentences):
    word_list = []
    for sent in sentences:
        for word in sent:
            word_list.append(word)
    return word_list

# 희귀 단어 제거 함수
def remove_rare_word(word_counts):
    threshold = 3
    
    rare_cnt = 0  # 등장 빈도 수가 threshold 보다 작은 단어의 개수
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도 수 총합
    rare_freq = 0 # 등장 빈도 수가 threshold보다 작은 단어의 등장 빈도 수의 총합.
    # 단어와 빈도수의 pair를 key와 value로 받는다.
    for key, value in word_counts.items():
        total_freq = total_freq + value
    
        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    return rare_cnt

# 정수 인코딩
def texts_to_sequences(tokenized_X_data, word_to_index):
    encoded_X_data = []
    for sent in tokenized_X_data:
        index_sequences = []
        for word in sent:
            try:
                index_sequences.append(word_to_index[word])
            except KeyError:
                index_sequences.append(word_to_index['<UNK>'])
        encoded_X_data.append(index_sequences)
    return encoded_X_data

# 패딩
def pad_sequences(sentences, max_len=19):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, :len(sentence)] = np.array(sentence)[:max_len]
    return features