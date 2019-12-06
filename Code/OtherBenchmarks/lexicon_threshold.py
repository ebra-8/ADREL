import os
import numpy as np
from sklearn import metrics


def loadData():
    trainPosFile = open("./dnm_data/MultiLingual/Ru7030/train/Forum_Ru_pos.txt", "r")
    trainNegFile = open("./dnm_data/MultiLingual/Ru7030/train/Forum_Ru_neg.txt", "r")


    Test=[]
    y=[]
    for l in trainPosFile:
        for i in range(2):
            Test.append(l)
            y.append(1)
    trainPosFile.close()
    for l in trainNegFile:
        Test.append(l)
        y.append(0)
    trainNegFile.close()
    
    valPosFile = open("./dnm_data/MultiLingual/Ru7030/val/Forum_Ru_pos.txt", "r")
    valNegFile = open("./dnm_data/MultiLingual/Ru7030/val/Forum_Ru_neg.txt", "r")
 

    Texts=[]
    y=[]
    for l in valPosFile:
        for i in range(1):
            Texts.append(l)
            y.append(1)
    valPosFile.close()
    for l in valNegFile:
        Texts.append(l)
        y.append(0)
    valNegFile.close()
    y = np.array(y)
    N_new=len(y)
    index=np.arange(N_new)
    np.random.shuffle(index)
    index=list(index)
    Texts=[Texts[i] for i in index]
    y=[y[i] for i in index]

    return Texts,y
def loadLexicons():
    lexicons=[]
    with open('Lexicon/Lexicon_ru.txt') as f:
        for line in f:
            lexicons.append(line.replace('\n',''))
    return lexicons
def countFrequencies(lexicons,Texts):
    fre_total=[]
    for test_i in Texts:
        fre_sen=[]
        for lexicon_i in lexicons:
            fre_sen.append(test_i.count(lexicon_i))
        fre_total.append(fre_sen)
    return np.sum(np.array(fre_total),axis=-1)


def main():
    Texts,y=loadData()
    lexicons=loadLexicons()

    fre_total=countFrequencies(lexicons,Texts)
    performance=[]
    for threshold in range(0,10,1):
        label_pred=np.zeros(fre_total.shape)
        label_pred[fre_total>threshold]=1
        acc=metrics.accuracy_score(y,label_pred)
        rec=metrics.recall_score(y,label_pred)
        prec=metrics.precision_score(y,label_pred)
        f1=metrics.f1_score(y,label_pred)
        performance.append([threshold,acc,prec,rec,f1])
    performance=np.array(performance)
    performance_sort=performance[performance[:, -1].argsort()]
    print(performance_sort[-1])
    #threshold=performance_sort[-1][0]
    threshold=1
    fold=5
    fold_size=int(len(fre_total)/fold)
    best_performance=[]
    for fold_i in range(0,len(fre_total),fold_size):
        fre_total_val=fre_total[fold_i:fold_i+fold_size]
        label_pred_val=np.zeros(fre_total_val.shape)
        label_pred_val[fre_total_val>threshold]=1
        y_val=y[fold_i:fold_i+fold_size]
        
        acc=metrics.accuracy_score(y_val,label_pred_val)
        rec=metrics.recall_score(y_val,label_pred_val)
        prec=metrics.precision_score(y_val,label_pred_val)
        f1=metrics.f1_score(y_val,label_pred_val)
        best_performance.append([threshold,acc,prec,rec,f1])
    print(np.array(best_performance))
    np.savetxt('lexicon.txt',np.array(best_performance),fmt='%.4f')
if __name__ == '__main__':
    main()