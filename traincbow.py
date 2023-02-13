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

batch_size = 50
num_epochs = 5

def softmax(z):
    z = z - np.max(z, axis=0)
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



def search(w2v, vec, w1, w2, w3):
    sim = 0.0
    index = 0

    omitted = [w1, w2, w3]
    indices = [val for val in w2v.word_index.values() if val not in omitted]

    for word in indices:
        v = w2v.W1[word]
        score = np.matmul(np.array(vec), np.array(v))/(norm(np.array(vec))*norm(np.array(v)))
        if score > sim: 
            sim = score
            index = word
    
    return index

def topSimilar(w2v, word):

    v1 = w2v.W1[w2v.word_index[word]]
    similarities = []
    for word_,index in w2v.word_index.items():
        vec = w2v.W1[w2v.word_index[word_]]
        similarities.append((index,np.matmul(np.array(vec), np.array(v1))/(norm(np.array(vec))*norm(np.array(v1)))))

    similarities.sort(reverse=True, key = lambda x: x[1])
    
    for i in range(10):
        print(similarities[i][1], w2v.index_word[similarities[i][0]])




def evaluate(w2v):
    score = 0
    cnt = 0

    man = w2v.W1[w2v.word_index["man"]]
    woman = w2v.W1[w2v.word_index["woman"]]
    king = w2v.W1[w2v.word_index["king"]]
    queen = w2v.W1[w2v.word_index["queen"]]
    sim1 = np.matmul(np.array(man), np.array(king))/(norm(np.array(man))*norm(np.array(king)))
    sim2 = np.matmul(np.array(woman), np.array(queen))/(norm(np.array(woman))*norm(np.array(queen)))

    print("sim1 and sim2 are {}, {}".format(sim1, sim2))
    pred = woman-man+king
    sim2 = np.matmul(np.array(pred), np.array(queen))/(norm(np.array(pred))*norm(np.array(queen)))
    print("cosine of pred and queen: {}".format(sim2))

    
    for key,value in w2v.analogy_mappings.items():
        w1, w2 = key.split(":")[0], key.split(":")[1]
        w3, w4 = value.split(":")[0], value.split(":")[1]

        if w1 not in w2v.word_index or w2 not in w2v.word_index or w3 not in w2v.word_index or w4 not in w2v.word_index:
            print("word not present")
            continue

        w1vec, w2vec = w2v.W1[w2v.word_index[w1]], w2v.W1[w2v.word_index[w2]]
        w3vec = w2v.W1[w2v.word_index[w3]]

        # if cnt == 2:
        #     print(w1vec)
        
        pred = w2vec - w1vec + w3vec
        index = search(w2v, pred, w2v.word_index[w1], w2v.word_index[w2], w2v.word_index[w3])
        actual = w2v.word_index[w4]

        if actual == index: score += 1
        cnt += 1

        print("Actual: {}:{}::{}:{}, pred: {}".format(w1, w2, w3, w4, w2v.index_word[index]))
    
    acc = score/cnt
    print("Accuracy is {}".format(acc))

    cnt = 0
    score = 0
    for key,value in w2v.validation_mappings.items():
        w1, w2 = key.split(":")[0], key.split(":")[1]
        w3, w4 = value.split(":")[0], value.split(":")[1]

        if w1 not in w2v.word_index or w2 not in w2v.word_index or w3 not in w2v.word_index or w4 not in w2v.word_index:
            print("word not present")
            continue

        w1vec, w2vec = w2v.W1[w2v.word_index[w1]], w2v.W1[w2v.word_index[w2]]
        w3vec = w2v.W1[w2v.word_index[w3]]

        pred = w2vec - w1vec + w3vec
        index = search(w2v, pred, w2v.word_index[w1], w2v.word_index[w2], w2v.word_index[w3])
        actual = w2v.word_index[w4]

        if actual == index: score += 1
        cnt += 1

        print("Actual: {}:{}::{}:{}, pred: {}".format(w1, w2, w3, w4, w2v.index_word[index]))
    
    acc = score/cnt
    print("Accuracy is {}".format(acc))

def train_testmodel(w2v, sents, word_to_index, num_epochs):
    loss = 0.0
    step = 0

    batch_x = []
    batch_y = []
    for epoch in tqdm(range(num_epochs)):
        
        for m in tqdm(range(0,len(sents))):
            sent = sents[m]

            if len(sent) < (w2v.window_size*2 + 1):
                continue
            
            for i in range(w2v.window_size, len(sent) - w2v.window_size):
                train_word = word_to_one_hot_vector(sent[i], word_to_index, len(word_to_index.keys()))
                context_words = sent[i-w2v.window_size:i] + sent[i+1:i+w2v.window_size+1]
                # if sent[i] == "rajasthan":
                #     print("train word: {}".format(sent[i]))
                #     print("context words: {}".format(context_words))
                # continue

                if len(context_words) > 0:
                    y = train_word
                    x = context_words_to_vector(context_words, word_to_index)
                    batch_y.append(y)
                    batch_x.append(x)

                if len(batch_x) == batch_size:
                    step += 1
                    loss = w2v.train(step, batch_x, batch_y)
                    batch_x = []
                    batch_y = []
                    if step % 20 == 0: print("step: {}, loss: {}, epoch: {}".format(step, loss, epoch))
                    if step % 400 == 0:
                        evaluate(w2v)
                    
                    if step%1000 == 0:
                        print("saving weights")
                        with open('w2vecbow_v4.pkl', 'wb') as f:
                            pickle.dump(w2v, f)
    
    if len(batch_x) > 0:
        loss = w2v.train(step, batch_x, batch_y)
        print("step: {}, loss: {}".format(step+1, loss))


def count_instances(w2v, sents):
    
    total = 0

    for m in range(0,len(sents)):
        sent = sents[m]
        sent = preprocessing(sent)

        if len(sent) < (w2v.window_size*2 + 1):
            continue
        
        for i in range(w2v.window_size, len(sent) - w2v.window_size):
            context_words = sent[i-w2v.window_size:i] + sent[i+1:i+w2v.window_size+1]
            if len(context_words) > 0:
                total += 1
            
    print("Total instances are {}".format(total))

def exists(tokens, sents):

    for sent in sents:
        str1 = " ".join(sent)
        str2 = " ".join(tokens)

        if str1 == str2:
            return True
    
    return False


def main():

    # collecting all the sentences
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
    
    word_to_index, index_to_word = index_word_maps(words)


    # w2v = word2vec()
    # w2v.initialize(len(word_to_index.keys()), words, word_to_index, validation_mappings, analogy_mappings, index_to_word)
    # load w2vec model
    with open("./w2vecbow_v4.pkl", "rb") as f:
        w2v = pickle.load(f)
    print("Length of vocab is {}".format(w2v.V))

    # compute count of data instances currently present
    count_instances(w2v, sents)

    train_testmodel(w2v, sents, word_to_index, num_epochs)

    # print top ten similar words 
    #topSimilar(w2v, "india")
    
    # save weights
    with open('w2vecbow_v4.pkl', 'wb') as f:
        pickle.dump(w2v, f)

if __name__ == "__main__":
    main()