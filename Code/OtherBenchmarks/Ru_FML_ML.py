'''
Created on Jun 28, 2017

@author: eb

This is after talking to Jiaheng and one feature space for both languages (i.e. one tokenizer)
'''
from keras import backend as K
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)
    set_session(tf.Session(config=config))

import numpy
import random
numpy.random.seed(7)

from keras.layers import Input, Dense
from keras.layers import Flatten

from keras.models import Model
from keras.layers import LSTM,GRU
from keras.layers.convolutional import UpSampling2D, Conv1D, MaxPooling1D

from keras.layers.wrappers import Bidirectional
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers

from keras.utils import plot_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import myPerf
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

################ French Data Preparation
################
#### Build train data (list of texts) - 380


trainPosFile_Fr = open("./dnm_data/MultiLingual/Rus7030/train/DNM-Rus-pos.txt", "r")
trainNegFile_Fr = open("./dnm_data/MultiLingual/Rus7030/train/DNM-Rus-neg.txt", "r")

trainTexts_Fr=[]
y_train_Fr=[]
for l in trainPosFile_Fr:
    trainTexts_Fr.append(l)
    y_train_Fr.append(1)
trainPosFile_Fr.close()
for l in trainNegFile_Fr:
    trainTexts_Fr.append(l)
    y_train_Fr.append(0)

trainNegFile_Fr.close()
 
print (trainTexts_Fr[0])
print ('train set size is: ' +str(len(trainTexts_Fr)))
 
##Build the labels (Pos=1, Neg=0)

print ('The size of training labels is: ' + str(len(y_train_Fr)))
y_train_Fr = numpy.array(y_train_Fr) 
 
#### Build validation (test) data  - 55 + 117 = 172

valPosFile_Fr = open("./dnm_data/MultiLingual/Rus7030/val/DNM-Rus-pos.txt", "r")
valNegFile_Fr = open("./dnm_data/MultiLingual/Rus7030/val/DNM-Rus-neg.txt", "r")
 


valTexts_Fr=[]
y_val_Fr=[]
for l in valPosFile_Fr:
    valTexts_Fr.append(l)
    y_val_Fr.append(1)
valPosFile_Fr.close()
for l in valNegFile_Fr:
    valTexts_Fr.append(l)
    y_val_Fr.append(0)
valNegFile_Fr.close()
 
print ('validation set size is: ' +str(len(valTexts_Fr)))
 
print ('The size of validation labels is: ' + str(len(y_val_Fr)))
y_val_Fr = numpy.array(y_val_Fr)
 
# Build an indexed sequence for each repument
vocabSize_Fr = 20000
 
tokenizer_Fr = text.Tokenizer(num_words=vocabSize_Fr,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ",
                   char_level=False)
 
# Build the word to int mapping based on the training data
tokenizer_Fr.fit_on_texts(trainTexts_Fr)
 
# Build the sequence (Keras built-in) for LSTM
trainTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(trainTexts_Fr)
print (trainTextsSeq_Fr[0])
print (len(trainTextsSeq_Fr))
trainTextsSeq_Fr = numpy.array(trainTextsSeq_Fr)

 
valTextsSeq_Fr = tokenizer_Fr.texts_to_sequences(valTexts_Fr)
print (valTextsSeq_Fr[0])
print (len(valTextsSeq_Fr))
valTextsSeq_Fr = numpy.array(valTextsSeq_Fr)
 
# for non-sequence vectorization such as tfidf --> SVM
trainVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(trainTextsSeq_Fr, mode='tfidf')
#print (trainVecMatrix)
print ('training vector length: '+str(len(trainVecMatrix_Fr)))
print ('training vector columns: '+str(len(trainVecMatrix_Fr[0])))
 
valVecMatrix_Fr = tokenizer_Fr.sequences_to_matrix(valTextsSeq_Fr, mode='tfidf')
print ('validation vector length: '+str(len(valVecMatrix_Fr)))
print ('validation vector columns: '+str(len(valVecMatrix_Fr[0])))


################ English Data Preparation
#### Build train data (list of texts)

trainPosFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-pos.txt", "r")
trainNegFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-train-neg.txt", "r")
 

trainTexts=[]
y_train=[]
for l in trainPosFile:
    trainTexts.append(l)
    y_train.append(1)
trainPosFile.close()
for l in trainNegFile:
    trainTexts.append(l)
    y_train.append(0)
trainNegFile.close()
 
print (trainTexts[0])
print ('train set size is: ' +str(len(trainTexts)))
 
##Build the labels (Pos=1, Neg=0)

            
print ('The size of training labels is: ' + str(len(y_train)))
y_train = numpy.array(y_train)
 
#### Build unlabeled data - 58893

 
#### Build validation (test) data  - 132

valPosFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-pos.txt", "r")
valNegFile = open("./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test-neg.txt", "r")
 
valTexts=[]
for l in valPosFile:
    valTexts.append(l)
valPosFile.close()
for l in valNegFile:
    valTexts.append(l)
valNegFile.close()
 
print ('validation set size is: ' +str(len(valTexts)))
 
y_val=[]
with open('./dnm_data/MultiLingual/Setting1_trainWithExpert/DNM-test.cat','r') as f:
    for line in f:
        if line.strip() == "pos":
            y_val.append(1)
        elif line.strip() == "neg":
            y_val.append(0)
        else:
            print('label')
            exit()
print ('The size of validation labels is: ' + str(len(y_val)))
y_val = numpy.array(y_val)
 
 
# Build an indexed sequence for each repument
vocabSize_Eng = 20000
tokenizer_Eng = text.Tokenizer(num_words=vocabSize_Eng,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                   lower=True,
                   split=" ",
                   char_level=False)
# Build the word to int mapping based on the training data
tokenizer_Eng.fit_on_texts(trainTexts)

# Build the sequence (Keras built-in) for LSTM
trainTextsSeq = tokenizer_Eng.texts_to_sequences(trainTexts)
print (trainTextsSeq[0])
print (len(trainTextsSeq))
trainTextsSeq = numpy.array(trainTextsSeq)

 
valTextsSeq = tokenizer_Eng.texts_to_sequences(valTexts)
print (valTextsSeq[0])
print (len(valTextsSeq))
valTextsSeq = numpy.array(valTextsSeq)
 
# for non-sequence vectorization such as tfidf --> SVM
trainVecMatrix = tokenizer_Eng.sequences_to_matrix(trainTextsSeq, mode='tfidf')
#print (trainVecMatrix)
print ('training vector length: '+str(len(trainVecMatrix)))
print ('training vector columns: '+str(len(trainVecMatrix[0])))
 
valVecMatrix = tokenizer_Eng.sequences_to_matrix(valTextsSeq, mode='tfidf')
print ('validation vector length: '+str(len(valVecMatrix)))
print ('validation vector columns: '+str(len(valVecMatrix[0])))



class_weight = {0: 1.,1: 1}

N_train=len(trainTextsSeq_Fr)
TextsSeq_E=[]
Y_E=[]
temp_i=0
for i in range(N_train):
  temp_i+=1
  if y_train_Fr[i]==1:
    for j in range(1):
      TextsSeq_E.append(trainTextsSeq_Fr[i])
      Y_E.append(1)
  elif y_train_Fr[i]==0:
    if temp_i%1==0:
      TextsSeq_E.append(trainTextsSeq_Fr[i])
      Y_E.append(0)
    else:
      pass
  
  else:
    print('error')
    exit()

N_val=len(valTextsSeq_Fr)


temp_i=0
for i in range(N_val):
  if y_val_Fr[i]==1:
    for j in range(1):
      TextsSeq_E.append(valTextsSeq_Fr[i])
      Y_E.append(1)
  elif y_val_Fr[i]==0:
    if temp_i%1==0:
      TextsSeq_E.append(valTextsSeq_Fr[i])
      Y_E.append(0)
    else:
      pass
      #trainTextsSeq_E.append(valTextsSeq[i])
      #train_Y_E.append(0)
    temp_i+=1
  else:
    print('error')
    exit()


N_new=len(Y_E)
index=numpy.arange(N_new)
numpy.random.shuffle(index)
index=list(index)
TextsSeq_E=[TextsSeq_E[i] for i in index]
Y_E=[Y_E[i] for i in index]
TextsSeq_E=TextsSeq_E
Y_E=Y_E

fold=5
fold_size=int(N_new/fold)

folds=5
result_cv=[]
each_fold=0


for fold_i in range(0,N_new,fold_size):

    trainVecMatrix=TextsSeq_E[:fold_i]+ TextsSeq_E[fold_size+fold_i:]
    valVecMatrix=TextsSeq_E[fold_i:fold_i+fold_size]

    y_train=Y_E[:fold_i]+Y_E[fold_size+fold_i:]
    y_val=Y_E[fold_i:fold_i+fold_size]
    

    if len(numpy.unique(numpy.array(y_val)))==1:
        continue

    y_val=numpy.array(y_val)
    y_train=numpy.array(y_train)

    max_rep_length = 100
    trainVecMatrix = sequence.pad_sequences(trainVecMatrix, maxlen=max_rep_length)
    valVecMatrix = sequence.pad_sequences(valVecMatrix, maxlen=max_rep_length)

    trainTextsSeq = sequence.pad_sequences(trainTextsSeq, maxlen=max_rep_length)
    valTextsSeq = sequence.pad_sequences(valTextsSeq, maxlen=max_rep_length)

    ratio_test=0.5
    temp_train=numpy.concatenate((trainTextsSeq,y_train.reshape((-1,1))),axis=-1)

    unq, unq_idx = numpy.unique(temp_train[:, -1], return_inverse=True)
    unq_cnt = numpy.bincount(unq_idx)
    cnt = numpy.max(unq_cnt)
    out = numpy.empty((cnt+int(cnt*ratio_test),) + temp_train.shape[1:], temp_train.dtype)

    indices = numpy.random.choice(numpy.where(unq_idx==0)[0], cnt)
    out[0:cnt] = temp_train[indices]
    indices = numpy.random.choice(numpy.where(unq_idx==1)[0], int(ratio_test*cnt))
    out[cnt:int((1+ratio_test)*cnt)] = temp_train[indices]

    numpy.random.shuffle(out)
    selected=int(0.4*len(out))
    trainTextsSeq=out[:selected,:-1]
    y_train=out[:selected,-1]
    
    print('Train...')
    print(y_train.shape)
    print(y_val.shape)
    #print(trainVecMatrix.shape)
    #print(valVecMatrix.shape)

    #
    clf = RandomForestClassifier(n_estimators=1200, max_depth=130, random_state=12)
    #clf = svm.SVC(kernel='linear', C=1e-3, probability=True)
    #clf = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    #clf = KNeighborsClassifier(n_neighbors=3)
    #clf = GaussianNB()
    clf.fit(trainVecMatrix, y_train)


    #### LSTM
    #label_pr_p = model.predict(valTextsSeq)
    label_pr_p = clf.predict(valVecMatrix)
    auc=metrics.roc_auc_score(y_val,label_pr_p)

    #print(label_pr_p)
    performance=[]
    for threshold in numpy.arange(0.4,0.6,0.01):
        label_pr=[]
        for i in range(len(label_pr_p)):
            if label_pr_p[i]>threshold:
                label_pr.append(1)
            else:
                label_pr.append(0)

        label_pr=numpy.array(label_pr)

        #label_pr = model.predict_classes(valTextsSeq)

        acc=metrics.accuracy_score(y_val,label_pr)
        rec=metrics.recall_score(y_val,label_pr)
        prec=metrics.precision_score(y_val,label_pr)
        f1=metrics.f1_score(y_val,label_pr)
          #print(label_pr)
          #print(y_val)
        performance.append([threshold,acc,prec,rec,f1,auc])
    performance=numpy.array(performance)

    performance_sort=performance[performance[:, -2].argsort()]

    for e in performance_sort:
        print (e)

    label_pr_p = clf.predict_proba(valVecMatrix)
    label_pr_p = label_pr_p[:,1]
    label_pr = clf.predict(valVecMatrix)
    auc=metrics.roc_auc_score(y_val,label_pr_p)
    auc2=metrics.average_precision_score(y_val,label_pr_p)
    acc=metrics.accuracy_score(y_val,label_pr)
    rec=metrics.recall_score(y_val,label_pr)
    prec=metrics.precision_score(y_val,label_pr)
    f1=metrics.f1_score(y_val,label_pr)
    #print(label_pr)
    #print(y_val)
    result_cv.append(performance_sort[-1])


numpy.savetxt('./FMT_Ru_ML.txt',numpy.array(result_cv),fmt='%.4f')
#numpy.savetxt('results_performance/%s_1LSTM.txt'%lang,numpy.array(result_cv),fmt='%.4f')
result_cv = (numpy.mean(numpy.array(result_cv),axis=0))

print(result_cv)
