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

def search(model, vec, w1, w2, w3):
    sim = 0.0
    index = ""

    omitted = [w1, w2, w3]
    words_dict = model.wv.key_to_index
    words = [word for word in words_dict if word not in omitted]

    for word in words:
        v = model.wv[word]
        score = np.matmul(np.array(vec), np.array(v))/(norm(np.array(vec))*norm(np.array(v)))
        if score > sim: 
            sim = score
            index = word
    
    return index

def evaluate(model, analogy_mappings, validation_mappings):
    score = 0
    cnt = 0

    words_dict = model.wv.key_to_index

    for key,value in analogy_mappings.items():
        w1, w2 = key.split(":")[0], key.split(":")[1]
        w3, w4 = value.split(":")[0], value.split(":")[1]

        if w1 not in words_dict or w2 not in words_dict or w3 not in words_dict or w4 not in words_dict:
            print("word not present")
            continue

        w1vec, w2vec = model.wv[w1], model.wv[w2]
        w3vec = model.wv[w3]

        # if cnt == 2:
        #     print(w1vec)
        
        pred = w2vec - w1vec + w3vec
        pred_word = search(model, pred, w1, w2, w3)

        if w4 == pred_word: score += 1
        cnt += 1

        print("Actual: {}:{}::{}:{}, pred: {}".format(w1, w2, w3, w4, pred_word))
    
    acc = score/cnt
    print("Accuracy is {}".format(acc))

    cnt = 0
    score = 0
    
    for key,value in validation_mappings.items():
        w1, w2 = key.split(":")[0], key.split(":")[1]
        w3, w4 = value.split(":")[0], value.split(":")[1]

        if w1 not in words_dict or w2 not in words_dict or w3 not in words_dict or w4 not in words_dict:
            print("word not present")
            continue

        w1vec, w2vec = model.wv[w1], model.wv[w2]
        w3vec = model.wv[w3]

        pred = w2vec - w1vec + w3vec
        pred_word = search(model, pred, w1, w2, w3)

        if w4 == pred_word: score += 1
        cnt += 1

        print("Actual: {}:{}::{}:{}, pred: {}".format(w1, w2, w3, w4, pred_word))
    
    acc = score/cnt
    print("Accuracy is {}".format(acc))

def index_word_maps(data: list) -> tuple:
    
    words = sorted(list(set(data)))
    
    word_to_index = {word: index for index, word in enumerate(words)}
    index_to_word = {index: word for index, word in enumerate(words)}
    return word_to_index, index_to_word

sents = []
for f in os.listdir("./data/"):
    path = "./data/{}".format(f)
    file = open(path)
    file_sents = file.readlines()
    
    cnt = 0
    for sent in file_sents:
        tokens = sent.split()
        tokens = preprocessing(tokens)
        if len(tokens) > 25 and len(tokens) < 6: continue
        if tokens not in sents:
            sents.append(tokens)
            cnt += 1
        if cnt > 500: break

f = open("./gutenberg.txt")
lines = f.readlines()
for sent in lines:
    sents.append(sent.replace("\n", "").split())
f.close()
random.shuffle(sents)

# collect all the words
words = []
for sent in sents:
    words += sent

f = open("Analogy_dataset.txt")
lines = f.readlines()
analogy_mappings = {}
analogy_words = {}
for line in lines:
    words_ = line.split()
    words_ = [word.lower() for word in words_]
    analogy_mappings[words_[0]+":"+words_[1]] = words_[2]+":"+words_[3]
    analogy_words[words_[0]] = 1
    analogy_words[words_[1]] = 1
    analogy_words[words_[2]] = 1
    analogy_words[words_[3]] = 1
f.close()

f = open("Validation.txt")
lines = f.readlines()
validation_mappings = {}
validation_words = {}
for line in lines:
    words_ = line.split()
    words_ = [word.lower() for word in words_]
    validation_mappings[words_[0]+":"+words_[1]] = words_[2]+":"+words_[3]
    validation_words[words_[0]] = 1
    validation_words[words_[1]] = 1
    validation_words[words_[2]] = 1
    validation_words[words_[3]] = 1
f.close()

analogy_words = list(analogy_words.keys())
validation_words = list(validation_words.keys())

# add analogy and validation words to words list
for word in analogy_words: words.append(word.lower())
for word in validation_words: words.append(word.lower())

model = Word2Vec(sentences=sents, vector_size=150, window=5, min_count=1, workers=4, sg=0, cbow_mean=1, epochs=20)

evaluate(model, analogy_mappings, validation_mappings)
# sims = model.wv.most_similar('kathmandu', topn=10)
# print(sims)