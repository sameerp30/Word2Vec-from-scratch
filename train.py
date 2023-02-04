import numpy as np
import string
from nltk.corpus import stopwords
import math
from nltk.corpus import gutenberg
import nltk

def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z

class word2vec(object):

    def __init__(self):
        self.N = 300
        self.X_train = []
        self.y_train = []
        self.window_size = 2
        self.alpha = 0.0003
        self.words = []
        self.word_index = {}
        self.loss = 0

    def initialize(self,V,data):
        self.V = V
        self.W1 = np.random.uniform(-1, 1, (self.V, self.N))
        self.W2 = np.random.uniform(-1, 1, (self.N, self.V))

        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i
  
      
    def feed_forward(self, X):
        X = X.reshape(self.V,1)
        self.h = np.dot(self.W1.T, X)
        self.u = np.dot(self.W2.T, self.h)
        self.y_hat = softmax(self.u)
        return self.y_hat
          
    def backpropagate(self, x, t):
        t = t.reshape(self.V,1)
        x = x.reshape(self.V,1)
        e = self.y_hat - t
        self.grad_W2 = np.dot(self.h, e.T)
        self.grad_W1 = np.dot(x, np.dot(self.W2, e).T)
        self.W1 = self.W1 - self.alpha * self.grad_W1
        self.W2 = self.W2 - self.alpha * self.grad_W2
    

    def train(self, x, y):
        self.y_hat = self.feed_forward(x)
        self.backpropagate(x, y)
        
        index = list(y).index(1)
        loss = -1*self.u[index][0] + np.log(np.sum(np.exp(self.u)))

        return loss


def train_model(w2v, fields, word_to_index):
    num_epochs = 1
    loss = 0.0
    step = 0
    batch_size = 10
    total = 0

    for epoch in range(num_epochs):
        for article in fields:
            training_data = gutenberg.sents(article)
            num_sents = len(training_data)
            print(num_sents)

            for m in range(0,len(training_data)):
                sent = training_data[m]
                sent = preprocessing(sent)

                for i in range(0,len(sent)):
                    context_words = []
                    train_word = word_to_one_hot_vector(sent[i],word_to_index,len(word_to_index.keys()))
                    
                    for j in range(i-w2v.window_size,i+w2v.window_size):
                            if i!=j and j>=0 and j<len(sent):
                                context_words.append(sent[j])
                    
                    if len(context_words) > 0:
                        y = train_word
                        x = context_words_to_vector(context_words,word_to_index)
                        loss = w2v.train(x, y)
                        step += 1
                    
                    if step % 1 == 0: print("step: {}, loss: {}, epoch: {}".format(step, loss, epoch))



def preprocessing(sentence):
    stop_words = set(stopwords.words('english'))   
    x = [word.strip(string.punctuation) for word in sentence
                                     if word not in stop_words]
    x = [word for word in x if len(word) != 0]
    x = [word.lower() for word in x]
    
    return x


def index_word_maps(data: list) -> tuple:
    
    words = sorted(list(set(data)))
    
    word_to_index = {word: index for index, word in enumerate(words)}
    index_to_word = {index: word for index, word in enumerate(words)}
    return word_to_index, index_to_word


def word_to_one_hot_vector(word: str, word_to_index: dict, vocabulary_size: int) -> np.ndarray:
    one_hot_vector = np.zeros(vocabulary_size)
    one_hot_vector[word_to_index[word]] = 1
    return one_hot_vector


def context_words_to_vector(context_words: list,
                            word_to_index: dict) -> np.ndarray:
    vocabulary_size = len(word_to_index)
    context_words_vectors = [
        word_to_one_hot_vector(word, word_to_index, vocabulary_size)
        for word in context_words]
    return np.mean(context_words_vectors, axis=0)
        


def main():
    fields = nltk.corpus.gutenberg.fileids()
    
    words = []
    for field in fields:
        words += list(gutenberg.words(field))
    stop_words = set(stopwords.words('english') + ["[","]","''"])
    words = [word.strip(string.punctuation) for word in words
                                        if word not in stop_words and len(word)!=0]
    words = [word.lower() for word in words]
    word_to_index, index_to_word = index_word_maps(words)

    w2v = word2vec()
    w2v.initialize(len(word_to_index.keys()), words)

    train_model(w2v, fields, word_to_index)


    


if __name__ == "__main__":
    main()