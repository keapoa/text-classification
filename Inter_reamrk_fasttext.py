#！/root/anaconda3/bin/python3.6
# coding: utf-8
import pandas as pd
import numpy as np
import random
import jieba 
import jieba.analyse
import warnings
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import fasttext

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

print('读取数据')

train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
#test_a = pd.read_csv('test.csv')

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

print('***************************finish****************************')

