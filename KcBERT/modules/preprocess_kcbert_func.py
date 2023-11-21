#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
def max_length(text, encoded_text):
    """finding max_length sentnece

    Args:
        text (str): text list
        encoded_text (int): encoding text list through tokenizer (not padding, just encoding)

    Returns:
        str: max_length sentence
    """
    max_length_index = max(range(len(encoded_text)), key = lambda i : len(encoded_text[i]))
    max_length_sentence = text[max_length_index]
    return max_length_sentence

# tokenized_inputs = tokenizer(train_texts)
# max_length_sentence = max_length(train_texts, tokenized_inputs['input_ids'])

def mean_length(encoded_text):
    """finding mean_length of sentences

    Args:
        encoded_text (int): encoding text list through tokenizer (not padding, just encoding, numpy)

    Returns:
        int: mean_length 
    """
    mean_length = sum(map(len, encoded_text))/len(encoded_text)
    
    return mean_length

def below_threshold_len(max_len, sentence_list):
    """setting proper max length

    Args:
        max_len (int): arbitary max_length
        sentence_list (int): encoded text list 
    """
    count = 0
    for sent in sentence_list:
        if (len(sent) <= max_len):
            count += 1
    print('전체 샘플 중 길이가 {} 이하인 샘플의 비율: {}'.format(max_len, round((count/len(sentence_list))*100 ,2)))

#below_threshold_len(25, tokenized_inputs['input_ids'])

# import matplotlib.pyplot as plt
# plt.hist([len(text) for text in sentence_list], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

def find_max_length_with_threshold(threshold_percentage, tokenizer, texts):
    """determining max length with threshold

    Args:
        threshold_percentage (int): threshold
        tokenizer (_type_): tokenizer
        texts (str): texts list

    Returns:
        int: chossen max length
    """
    
    tokenized_inputs = tokenizer(texts)  # padding 이전 '정수 인코딩'
    sentenced_list = tokenized_inputs['input_ids']
    
    total_samples = len(sentenced_list)
    
    sorted_lengths = sorted(map(len, sentenced_list))
    
    threshold_index = int(threshold_percentage / 100 * total_samples) - 1
    max_len = sorted_lengths[threshold_index]
    
    return max_len

#max_len = find_max_length_with_threshold(99, tokenized_inputs['input_ids'])
        