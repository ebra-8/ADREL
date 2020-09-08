import numpy
import h5py
numpy.random.seed(7)

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras import optimizers

from glove import Corpus, Glove

from keras.utils import plot_model

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import myPerf


################ Russian Data Preparation
######### GloVe Word Representation Training for Russian
corpus_en = []
with open('/home/eb/GloVe/corpus_ru','r') as f:
    for line in f:
        corpus_en.append(line)

new_corpus_en=[]
for line in corpus_en:    
    splited = line.split()
    new_corpus_en.append(splited)
corpus_en=new_corpus_en
corpus = Corpus() 
corpus.fit(corpus_en, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove_ru.model')
glove.add_dictionary(corpus.dictionary)


######## Obtain initial representations from the trained GloVe
embeddings_dict={}
for key in glove.dictionary:
    embeddings_dict[key] = glove.word_vectors[glove.dictionary[key]]
    
print ('GloVe representations learned for Russian. vocab size is: ' + str(len(embeddings_dict)))

#### Build train data (list of texts) - 380
trainPosFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-pos.txt", "r")
trainNegFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-Rus-neg.txt", "r")

trainTexts_Ru=[]
for l in trainPosFile:
    trainTexts_Ru.append(l)
trainPosFile.close()
for l in trainNegFile:
    trainTexts_Ru.append(l)
trainNegFile.close()

print trainTexts_Ru[0]
print ('train set size is: ' +str(len(trainTexts_Ru)))

##Build the labels (Pos=1, Neg=0)
y_train_Ru=[]
with open('/home/eb/dnm_data/MultiLingual/Rus7030/train/DNM-RUS-train.cat','r') as f:
    for line in f:
        if line.strip() == "pos":
            y_train_Ru.append(1)
        else:
            y_train_Ru.append(0)
print ('The size of training labels is: ' + str(len(y_train_Ru)))


#### Build validation (test) data  - 55 + 117= 172
valPosFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-pos.txt", "r")
valNegFile = open("/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-Rus-neg.txt", "r")

valTexts_Ru=[]
for l in valPosFile:
    valTexts_Ru.append(l)
valPosFile.close()
for l in valNegFile:
    valTexts_Ru.append(l)
valNegFile.close()

print ('validation set size is: ' +str(len(valTexts_Ru)))

y_val_Ru=[]
with open('/home/eb/dnm_data/MultiLingual/Rus7030/val/DNM-RUS-test.cat','r') as f:
    for line in f:
        if line.strip() == "pos":
            y_val_Ru.append(1)
        else:
            y_val_Ru.append(0)
print ('The size of validation labels is: ' + str(len(y_val_Ru)))

# Build an indexed sequence for each repument
vocabSize_Ru = 20000

tokenizer_Ru = text.Tokenizer(num_words=vocabSize_Ru,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ",
                   char_level=False)

# Build the word to int mapping based on the training data
tokenizer_Ru.fit_on_texts(trainTexts_Ru)

# Build the sequence (Keras built-in) for LSTM
trainTextsSeq_Ru = tokenizer_Ru.texts_to_sequences(trainTexts_Ru)
print (trainTextsSeq_Ru[0])
print (len(trainTextsSeq_Ru))

valTextsSeq_Ru = tokenizer_Ru.texts_to_sequences(valTexts_Ru)
print (valTextsSeq_Ru[0])
print (len(valTextsSeq_Ru))

# for non-sequence vectorization such as tfidf --> SVM
trainVecMatrix_Ru = tokenizer_Ru.sequences_to_matrix(trainTextsSeq_Ru, mode='tfidf')
#print (trainVecMatrix)
print ('training vector length: '+str(len(trainVecMatrix_Ru)))
print ('training vector columns: '+str(len(trainVecMatrix_Ru[0])))

valVecMatrix_Ru = tokenizer_Ru.sequences_to_matrix(valTextsSeq_Ru, mode='tfidf')
print ('validation vector length: '+str(len(valVecMatrix_Ru)))
print ('validation vector columns: '+str(len(valVecMatrix_Ru[0])))

################ English Data Preparation

######### GloVe Word Representation Training for English
corpus_en = []
with open('/home/eb/GloVe/corpus_en','r') as f:
    for line in f:
        corpus_en.append(line)
 
new_corpus_en=[]
for line in corpus_en:    
    splited = line.split()
    new_corpus_en.append(splited)
corpus_en=new_corpus_en
corpus = Corpus() 
corpus.fit(corpus_en, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove_en.model')
glove.add_dictionary(corpus.dictionary)
 
 
######## Obtain initial representations from the trained GloVe
embeddings_dict={}
for key in glove.dictionary:
    embeddings_dict[key] = glove.word_vectors[glove.dictionary[key]]
     
print ('GloVe representations learned for English. vocab size is: ' + str(len(embeddings_dict)))
