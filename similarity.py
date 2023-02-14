import numpy as np
import string
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
import nltk
import pickle
from numpy.linalg import norm
import random
from tqdm import tqdm
import argparse


def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z, axis = 0)
    return e_z / sum_e_z

class word2vec(object):

    def __init__(self):
        self.N = 100
        self.X_train = []
        self.y_train = []
        self.window_size = 4
        self.alpha = 0.09
        self.words = []
        self.word_index = {}
        self.validation_mappings = {}
        self.analogy_mappings = {}
        self.loss = 0

    def initialize(self,V, data, word_index, validation_mappings, analogy_mappings, index_word):
        self.V = V
        np.random.seed(1)
        self.W1 = np.random.rand(self.V, self.N)
        self.W2 = np.random.rand(self.N, self.V)

        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
        
        self.word_index = word_index
        self.index_word = index_word
        self.validation_mappings = validation_mappings
        self.analogy_mappings = analogy_mappings
  
      
    def feed_forward(self, X):
        # transposing X to shape V, batch_size
        X_ = np.array(X).T
        
        assert X_.shape == (self.V, len(X))
        
        self.h = np.matmul(self.W1.T, X_)
        self.u = np.matmul(self.W2.T, self.h)
        self.y_hat = softmax(self.u)
        
        assert self.y_hat.shape == (self.V, len(X))

        return self.y_hat
          
    def backpropagate(self, x, t):
        t_ = np.array(t).T
        x_ = np.array(x).T

        assert t_.shape == (self.V, len(x))
        assert x_.shape == (self.V, len(x))

        e = self.y_hat - t_
        self.grad_W2 = np.matmul(self.h, e.T)
        self.grad_W1 = np.matmul(x_, np.matmul(self.W2, e).T)
        self.W1 = self.W1 - self.alpha * self.grad_W1
        self.W2 = self.W2 - self.alpha * self.grad_W2
    

    def train(self, step, batch_x, batch_y):
        self.y_hat = self.feed_forward(batch_x)
        self.backpropagate(batch_x, batch_y)
        
        u = self.u.T
        u = np.array([u[i][list(y).index(1)] for i,y in enumerate(batch_y)])
        
        #print(u)
        loss = None
        if step % 20 == 0:
            loss = (-1*u.sum(axis=0) + np.log(np.sum(np.exp(self.u), axis=0)).sum())/len(batch_x)

        return loss

def isEnglish(word):
    hasNum = any(chr.isdigit() for chr in word)
    flag = False
    for char in word:
        if (char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z'):
            continue
        flag = True
        break

    return not (hasNum or flag)

def preprocessing(sentence):
    stop_words = set(stopwords.words('english'))
    
    # remove stopwords and punctuation
    x = [word.strip(string.punctuation).lower() for word in sentence]
    x = [word for word in x if word not in stop_words]

    # remove words containing numbers and non-english characters
    x = [word for word in x if isEnglish(word)]
    x = [word for word in x if len(word) != 0 and len(word) > 2]
    
    return x

def topSimilar(w2v, pred, k, w1, w2, w3):

    v1 = pred
    omit = [w1, w2, w3]
    indices = [val for key,val in w2v.word_index.items() if key not in omit]
    
    similarities = []
    for index in indices:
        vec = w2v.W1[index]
        similarities.append((index,np.matmul(np.array(vec), np.array(v1))/(norm(np.array(vec))*norm(np.array(v1)))))

    similarities.sort(reverse=True, key = lambda x: x[1])
    
    for i in range(k):
        print(similarities[i][1], w2v.index_word[similarities[i][0]])

def main():

    # load weights
    with open('w2vecbow_v4.pkl', 'rb') as f:
        w2v = pickle.load(f)

    # collect the input sentence
    parser = argparse.ArgumentParser()

    parser.add_argument("-i1", "--Input1", help = "Provide Input")
    parser.add_argument("-i2", "--Input2", help = "Provide Input")

    args = parser.parse_args()

    w1 = args.Input1
    w2 = args.Input2

    # pre-process words
    words = preprocessing([w1, w2])
    w1, w2 = words[0], words[1]

    if w1 not in w2v.word_index:
        print("{} not present in vocabulary".format(w1))
        exit() 
    if w2 not in w2v.word_index:
        print("{} not present in vocabulary".format(w2))
        exit()
    
    w1vec, w2vec = w2v.W1[w2v.word_index[w1]], w2v.W1[w2v.word_index[w2]]
    similarity = np.matmul(np.array(w1vec), np.array(w2vec))/(norm(np.array(w1vec))*norm(np.array(w2vec)))
    
    print(similarity)
    



if __name__ == "__main__":
    main()