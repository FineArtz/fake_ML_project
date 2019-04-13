# coding = utf-8
# 2019-04-04
# cut by single character


import codecs

f = codecs.open("handout\\train_shuffle.txt", "r", encoding="utf-8")
labelList = []
commentList = []
strList = f.readlines()
for str in strList:
    sepstr = str.split('\t')
    labelList.append(1 if sepstr[0] == '1' else 0)
    commentList.append(sepstr[1])
f.close()

fout = codecs.open("raw_text.txt", "w")
for ss in commentList:
    fout.write(ss.strip() + "\n")
fout.close()

fout = codecs.open("cut_text.txt", "w")
ffout = codecs.open("washed_text.txt", "w")
fout_p = codecs.open("positive.txt", "w")
fout_n = codecs.open("negative.txt", "w")
fout_l = codecs.open("label.txt", "w")
stopList = [w.strip() for w in codecs.open("stop_words.txt", "r", encoding="utf-8").readlines()]
i = 0
for ss in commentList:
    segList = [c for c in ss]
    fout.write("%d %s\n" % (labelList[i], " ".join(segList).strip()))
    washedList = []
    for word in segList:
        word = word.strip()
        if word not in stopList:
            washedList.append(word)
    ffout.write(" ".join(washedList).strip() + "\n")
    if labelList[i] == 1:
        fout_p.write(" ".join(washedList).strip() + "\n")
    else:
        fout_n.write(" ".join(washedList).strip() + "\n")
    fout_l.write("%d\n" % labelList[i])
    i = i + 1
fout.close()
ffout.close()
fout_p.close()
fout_n.close()
fout_l.close()
