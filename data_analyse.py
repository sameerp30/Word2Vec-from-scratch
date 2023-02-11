import os

def count_samanantar():
    for f in os.listdir("./samanantar/"):
        path = "./samanantar/{}".format(f)
        file = open(path)
        sents = file.readlines()
        
        cnt = 0
        for sent in sents:
            tokens = sent.split()
            if len(tokens) > 20 and len(tokens) < 6: continue
            cnt += 1
        
        print(path + ": {}".format(cnt))

def combine_files():
    paths2013 = os.listdir("data2013rem/")
    paths2012 = os.listdir("data2012/")

    #print(paths2012)

    for path in paths2013:
        path12 = path.replace("13", "12")
        if path == "news13_count.txt": continue
        if path12 in paths2012:
            file = open("data2013rem/{}".format(path))
            file1 = open("data2012/{}".format(path12))
            lines13 = file.readlines()
            lines12 = file1.readlines()


            lines = lines13 + lines12
            print(len(lines12), len(lines13), len(lines))

            file2 = open("data2012/{}".format(path12), "w+")
            for line in lines:
                line = line.replace("\n", "")
                file2.write(line + "\n")
            
            file.close()
            file1.close()
            file2.close()

def main():

    #count_samanantar()

    combine_files()



if __name__ == "__main__":
    main()