import nltk

sentence = 'NLTK is a leading platform for building Python programs to work with human language data. '
tokens = nltk.word_tokenize(sentence)  # 分词
tagged = nltk.pos_tag(tokens)  # 词性标注
entities = nltk.chunk.ne_chunk(tagged)  # 命名实体识别
print(tokens)
print(tagged)
print(entities)
