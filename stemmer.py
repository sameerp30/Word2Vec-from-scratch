import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

def isInflectional(tokens, test_token, index):
    base = ps.stem(test_token)
    print(base)
    for i, token in enumerate(tokens):
        if i == index: continue
        if token == base and token != test_token: return True
    
    return False



def check(sent):
    tokens = sent.split()
    flag = False

    for i, token in enumerate(tokens):
        if isInflectional(tokens, token, i):
            flag = True
            break
    
    return flag


print(check("think thinks"))