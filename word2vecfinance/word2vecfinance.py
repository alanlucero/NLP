
# This file uses the word2vec package installed via pip, i think it implements 
# the google word2vec directly

import word2vec

workdir = r'c:\\Users\\bungum\\c\\bloomberg'

trainfile = workdir+r'\all.w2v'

# words 2 phrases
word2vec.word2phrase(trainfile, 
                     workdir + r'\bloomberg-phrases', 
                     verbose=True)

# train model on phrases instead of orig
word2vec.word2vec(workdir + r'\bloomberg-phrases', 
                  workdir + r'\bloomberg.bin', 
                  size=100, 
                  verbose=True)

# do clustering on the words
word2vec.word2clusters(trainfile, 
                       workdir + r'\bloomberg-clusters.txt', 
                       100, 
                       verbose=True)


# load model
model = word2vec.load(workdir + r'\bloomberg.bin')



indexes, metrics = model.analogy(pos=['president'], neg=['united_states'], n=10)

indexes, metrics = model.cosine('precious_metals')
resplist = model.generate_response(indexes, metrics).tolist()

for elm in resplist:
    print(elm)