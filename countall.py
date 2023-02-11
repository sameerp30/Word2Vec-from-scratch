

dict = {}
count1 = open("data2012/news12_count.txt")
lines = count1.readlines()

for line in lines:
    key, count = line.split(":")[0], float(line.split(":")[1].replace("\n", ""))
    dict[key] = count

count1 = open("data2012rem/news12_count.txt")
lines = count1.readlines()

for line in lines:
    key, count = line.split(":")[0], float(line.split(":")[1].replace("\n", ""))
    dict[key] = count

count1 = open("data2013rem/news13_count.txt")
lines = count1.readlines()

for line in lines:
    key, count = line.split(":")[0], float(line.split(":")[1].replace("\n", ""))
    dict[key] += count

f = open("totalcounts.txt", "w+")
for key,count in dict.items():
    f.write(key + ":" + str(count) + "\n")


