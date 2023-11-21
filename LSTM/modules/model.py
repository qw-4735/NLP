#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class LSTM(nn.Module):
    def __init__(self, vocab_size, config):   # vocab_size :  전체 단어의 수
        super(LSTM, self).__init__()
        
        self.config = config
        
        self.embedding = nn.Embedding(vocab_size+2, config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True)
        self.fc = nn.Linear(config['hidden_dim'], config['output_dim'])
        
    def forward(self, input):  # input : (batch_size, seq_length)
        
        embedded = self.embedding(input)  # (batch_size, seq_length, embedding_dim)
        
        output, (hidden, cell) =  self.lstm(embedded)  # output : (batch_size, seq_length, hidden_dim), hidden : (1, batch_size, hidden_dim)
        last_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)  # (batch_size, output_dim)
        return logits