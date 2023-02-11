import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import string
from nltk.corpus import stopwords
import time

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

def scrape_news2012(word1, word2, lines):
    # lines has sentences in the form of list of tokens

    count1 = 0.0
    sent = []
    for sentence in lines:
        sentence = sentence.replace("\n", "")

        first=0.0
        second=0.0
        for word in sentence.split():
            if(word==word1):
                first=1
            # if(word==word2):
            #     second=1
            # if(first==1 and second==1):
            if(first==1):
                # f2 = open(f"./data2013rem/news13_{word1}_{word2}.txt", "a+")
                f2 = open(f"./data2013rem/news13_{word1}.txt", "a+")
                if sentence not in sent:
                    f2.write(sentence + "\n")
                    count1 = count1 + 1
                    sent.append(sentence)
                break
    
    f = open(f"./data2013rem/news13_count.txt", "a+")
    # f.write(word1+" "+word2+":"+str(count1)+"\n")
    f.write(word1+":"+str(count1)+"\n")
    
    return count1

file = open(f"./news.2013.en.shuffled.txt", "r")
lines = file.readlines()

# solos = ["krone", "dinar", "algeria", "usa", "dollar", "argentina", "peso", "russia", "ruble", "armenia", "dram", "sweden", "krona", 
#          "denmark", "nigeria", "naira", "mumbai", "maharashtra", "telangana", "hyderabad", "assam", "dispur", "tripura", "agartala", "slovakia", 
#          "slovakian"]
solos = ["gujarat", "gandhinagar"]

#print(scrape_news2012("gujarat", "gandhinagar", lines))

#with open("./analogies_remaining2.txt", "r") as file:
for word in solos:
    # Read the words into a list
    #words = file.read().split()
    #sentences = []
    #checked=[]
    t1 = time.time()
    print(word, scrape_news2012(word.lower(), None, lines))
        
    # for i in range(0, len(words), 2):
        # Call the function with the word you want to search for
        # if(words[i]+words[i+1] not in checked):
            # print(words[i], words[i+1], scrape_news2012(words[i].lower(), words[i+1].lower(), lines))
            # checked.append(words[i]+words[i+1])
    
    print(time.time() - t1)