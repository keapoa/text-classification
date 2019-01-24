#！/root/anaconda3/bin/python3.6
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import jieba 
import jieba.analyse
from gensim.models import word2vec 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import scale
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import warnings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import layers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model
from keras.models import Model,load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from snownlp import SnowNLP
from keras.utils import multi_gpu_model
import fasttext
import gc
import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import f1_score,accuracy_score
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning,module = 'gensim' )
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        #_val_recall = recall_score(val_targ, val_predict)
        #_val_precision = precision_score(val_targ, val_predict)
        #print('验证集recall为{}'.format(_val_recall))
        #print('验证集precision为{}'.format(_val_precision))
        print('验证集macro f1 {}'.format(_val_f1))
        if _val_f1 > self.best_f1:
            self.best_f1 = _val_f1
            self.model.save('bi_gru.h5')
        return

print('读取数据')

train = pd.read_csv('train.csv')
valid = pd.read_csv('data_sample.csv')
#test_a = pd.read_csv('sentiment_analysis_testa.csv')

words = [train,valid]
#生成训练文件
with open('train.txt','w') as f:
    for i in range(train.shape[0]):
        seg = ' '.join(jieba.cut(train['评论'][i]))
        out = seg + '\t__label__' + str(train['y'][i]) + '\n' 
        f.write(out)
f.close()
with open('valid.txt','w') as f:
    for i in range(valid.shape[0]):
        seg = ' '.join(jieba.cut(valid['评论'][i]))
        out = seg + '\t__label__' + str(valid['y'][i]) + '\n'
        f.write(out)
f.close()
print('文件生成完成')
print('开始训练模型')
classifier = fasttext.supervised("train.txt","news_fasttext.model",label_prefix="__label__")
#print('加载训练好的模型')
#classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
print('测试模型')
result = classifier.test("valid.txt")
        
print(result.precision)
print(result.recall)
print('预测生成模型')

val_text = [' '.join(jieba.cut(i)) for i in valid['评论']]

y_val_pred = [int(pre[0]) for pre in classifier.predict(val_text)]
a = classifier.predict(val_text)
with open('result.txt','w') as f:
    f.write(str(a))
f.close()
print(type(a[0][0]))
val_y = valid['y']
 
#y_val_pred = y_val_pred.tolist()
y_val_pred_1 = list(map(lambda x :x-1,y_val_pred))
y_val_pred_2 = list(map(lambda x :x-2,y_val_pred))
val_y_1 = list(map(lambda x :x-1,val_y))
val_y_2 = list(map(lambda x :x-2,val_y))
precision_0 = precision_score(list(map(lambda x: not x ,y_val_pred)),list(map(lambda x: not x ,val_y.values)))
precision_1 = precision_score(list(map(lambda x: not x ,y_val_pred_1)),list(map(lambda x: not x ,val_y_1)))
precision_2 = precision_score(list(map(lambda x: not x ,y_val_pred_2)),list(map(lambda x: not x ,val_y_2)))
recall_0 = recall_score(list(map(lambda x: not x ,y_val_pred)),list(map(lambda x: not x ,val_y.values)))
recall_1 = recall_score(list(map(lambda x: not x ,y_val_pred_1)),list(map(lambda x: not x ,val_y_1)))
recall_2 = recall_score(list(map(lambda x: not x ,y_val_pred_2)),list(map(lambda x: not x ,val_y_2)))
f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
f1 = (f1_0 + f1_1 + f1_2)/3
print('计算的precision_0值为{}'.format(precision_0))
print('计算的precision_1值为{}'.format(precision_1))
print('计算的precision_2值为{}'.format(precision_2))
    
print('计算的reacll_0值为{}'.format(recall_0))
print('计算的reacll_1值为{}'.format(recall_1))
print('计算的reacll_2值为{}'.format(recall_2))
    
print('计算的f1_0值为{}'.format(f1_0))
print('计算的f1_1值为{}'.format(f1_1))
print('计算的f1_2值为{}'.format(f1_2))
    
print('计算的macro f1值为{}'.format(f1))
    
F1_score = f1_score(y_val_pred, val_y, average='macro')
print('模型的macro f1为：{}'.format(F1_score))
#y_test_pred = np.argmax(y_test_pred, axis=1)


#out.to_csv('bi_gru_shuju1.csv',index = False)
print('***************************finish****************************')

