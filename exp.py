from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
import nltk
import pickle
from numpy.linalg import norm
import random
from tqdm import tqdm
import os


def preprocessing(sentence):
    stop_words = set(stopwords.words('english'))
    
    # remove stopwords and punctuation
    x = [word.strip(string.punctuation).lower() for word in sentence]
    x = [word for word in x if word not in stop_words]

    # remove words containing numbers and non-english characters
    x = [word for word in x if isEnglish(word)]
    x = [word for word in x if len(word) != 0 and len(word) > 2]
    
    return x

def isEnglish(word):
    hasNum = any(chr.isdigit() for chr in word)
    flag = False
    for char in word:
        if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
            continue
        flag = True
        break

    return not (hasNum or flag)

def exists(tokens, sents):

    for sent in sents:
        str1 = " ".join(sent)
        str2 = " ".join(tokens)

        if str1 == str2:
            return True
    
    return False

path = "./samanantar/newsamanantar_India_Delhi.txt"
file = open(path)
file_sents = file.readlines()
cnt = 0
sents = []
for sent in file_sents:
    tokens = sent.split()
    tokens = preprocessing(tokens)
    if len(tokens) > 20 and len(tokens) < 6: continue
    if not exists(tokens, sents):
        sents.append(tokens)
    cnt += 1
    #if cnt > 120: break
random.shuffle(sents)

model = Word2Vec(sentences=sents, vector_size=300, window=5, min_count=1, workers=4, sg=0, cbow_mean=1)
sims = model.wv.most_similar('india', topn=10)
print(sims)
# sims = model.wv.most_similar('kathmandu', topn=10)
# print(sims)