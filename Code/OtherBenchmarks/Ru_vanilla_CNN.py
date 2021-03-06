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


model_Name='CNN3'
for fold_i in range(0,N_new,fold_size):

    trainTextsSeq_Fr=TextsSeq_E[:fold_i]+ TextsSeq_E[fold_size+fold_i:]
    valTextsSeq_Fr=TextsSeq_E[fold_i:fold_i+fold_size]

    y_train_Fr=Y_E[:fold_i]+Y_E[fold_size+fold_i:]
    y_val_Fr=Y_E[fold_i:fold_i+fold_size]
    
    if len(numpy.unique(numpy.array(y_val_Fr)))==1:
        continue

    max_rep_length = 150

    trainTextsSeq_Fr = sequence.pad_sequences(trainTextsSeq_Fr, maxlen=max_rep_length)
    valTextsSeq_Fr = sequence.pad_sequences(valTextsSeq_Fr, maxlen=max_rep_length)

    ratio_test=0.5
    temp_train=numpy.concatenate((trainTextsSeq_Fr,y_train_Fr.reshape((-1,1))),axis=-1)

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
    trainTextsSeq_Fr=out[:selected,:-1]
    y_train_Fr=out[:selected,-1]
    
    embedding_vector_length = 100
    class_names=['0','1']

    # Building the model on the entire dataset and testing it on the test data
    runs=[]
    for run in range(1):
        myInput = Input(shape=(max_rep_length,), name='input_English') # 500 vectors of size 100
        x = Embedding(output_dim=100, input_dim=vocabSize_Fr, input_length=max_rep_length)(myInput)
        CNN_h1 = Conv1D(200, kernel_size=4, strides=1, activation='relu')(x)
        h1 = MaxPooling1D(pool_size=2, strides=None, padding='same')(CNN_h1)
        CNN_h2 = Conv1D(200,  kernel_size=3, strides=1, activation='relu')(h1)
        h2 = MaxPooling1D(pool_size=2, strides=None, padding='same')(CNN_h2)
        CNN_h3 = Conv1D(200,  kernel_size=3, strides=1, activation='relu')(h2)
        h3 = MaxPooling1D(pool_size=2, strides=None, padding='same')(CNN_h3)
        CNN_h4 = Conv1D(200,  kernel_size=3, strides=1, activation='relu')(h3)
        h5 = MaxPooling1D(pool_size=2, strides=None, padding='same')(CNN_h4)

        f = Flatten()(h1)
        predictions = Dense(2, activation='softmax', name='pred_eng')(f)
           
        model = Model(myInput, outputs=predictions)
        model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
           
        print ("input shape: " + str(numpy.shape(trainTextsSeq_Fr)))
        print ("y_train: " + str(numpy.shape(y_train_Fr)))
        print (y_train_Fr)
           
        model.fit(trainTextsSeq_Fr,y_train_Fr, epochs=10, batch_size=16)#, callbacks=[monitor])
           
        lstm_pred_EngFr = model.predict(valTextsSeq_Fr, batch_size=3, verbose=0)
           
        print (lstm_pred_EngFr)
        ####Get scores for AUC
        ####
        lstm_pred_EngFr=lstm_pred_EngFr[:,1]
        auc=metrics.roc_auc_score(y_val_Fr,lstm_pred_EngFr)

        performance=[]
        for threshold in numpy.arange(0.3,0.7,0.01): 
            all_predictions=numpy.zeros(len(y_val_Fr))
            all_predictions[lstm_pred_EngFr>threshold]=1

            acc=metrics.accuracy_score(y_val_Fr,all_predictions)
            rec=metrics.recall_score(y_val_Fr,all_predictions)
            prec=metrics.precision_score(y_val_Fr,all_predictions)
            f1=metrics.f1_score(y_val_Fr,all_predictions)
            performance.append([threshold,acc,prec,rec,f1,auc])

        performance=numpy.array(performance)

        performance_sort=performance[performance[:, -2].argsort()]

        print(performance_sort[-20:])
        
        with open("CNN/FMT_Ru_%s_add.txt"%model_Name, "ab") as f:
            #f.write(b"\n")
            numpy.savetxt(f, performance_sort[-1].reshape((-1,6)),fmt='%.4f')


    #save predictions
    #numpy.savetxt('/output/Pred_English.txt',lstm_pred_EngRus[0],'%.0f',delimiter=',')
    #numpy.savetxt('output/OurMethod_Pred_French.txt',lstm_pred_EngRus[1],'%.0f',delimiter=',')
    #numpy.savetxt('output/OurMethod_Pred_French_RUNS.txt',runs,'%.0f',delimiter=',')
      
    #plot_model(model, to_file='model.png')

