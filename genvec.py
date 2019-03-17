# encoding=utf-8
# 2019-03-17
# Generate word vectors

import codecs

fin_t = codecs.open("word_weight.txt", "r")
fin_p = codecs.open("positive.txt", "r")
fin_n = codecs.open("negative.txt", "r")
fout = codecs.open("word_vector.csv", "w")

tags = fin_t.readlines()
tagDict = {}
for tag in tags:
    tag = tag.strip().split(':')
    tagDict[tag[0]] = tag[1]

sentences = fin_p.readlines()
for sen in sentences:
    words = list(map(lambda s : s[0 : -1], sen.split(' ')))[0 : -1]
    wordvec = ["1"]
    for tag in tagDict:
        wordvec.append(tagDict[tag] if tag in words else "0")
    fout.write(",".join(wordvec) + '\n')

sentences = fin_n.readlines()
for sen in sentences:
    words = list(map(lambda s : s[0 : -1], sen.split(' ')))[0 : -1]
    wordvec = ["0"]
    for tag in tagDict:
        wordvec.append(tagDict[tag] if tag in words else "0")
    fout.write(",".join(wordvec) + '\n')
    
fin_t.close()
fin_p.close()
fin_n.close()
fout.close()
