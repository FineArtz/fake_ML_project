# encoding=utf-8
# 2019-03-17
# Generate TF-IDF weights

import jieba
import jieba.analyse
import codecs

fin = codecs.open("washed_text.txt", "rb")
content = fin.read()

jieba.analyse.set_stop_words("stop_words.txt")

tags = jieba.analyse.extract_tags(content, topK=None, withWeight=True)

fout = codecs.open("word_weight.txt", "w")
for tag in tags:
    fout.write("%s:%f\n" % (tag[0] if tag[0][-1] != "犇" else tag[0][:-1], tag[1]))

fin.close()
fout.close()
