
# coding: utf-8

# In[ ]:


import pandas as pd
from keras.models import Model,load_model
import json
import jieba

test = pd.read_csv('id_comment_pred.csv')

texts = [' '.join(jieba.cut(str(i))).split() for i in test['reviewContent']]
word_dict = json.load(open('wi.json','r'))
#当测试集未被训练时将词转化为索引
def word_index_map(texts,word_index):
    for text in texts:
        for i in range(len(text)):
            text[i] = word_index[text[i]]
    return texts
text_data = word_index_map(texts,word_index)
#当测试集有未登录词时 - 替换为 -unknow
def unk_word_replace(texts,word_index):
    for text in texts:
        for i in range(len(text)):
            if text[i] not in word_index:
                text[i] = word_index['unk']
            else:
                text[i] = word_index[text[i]]
    return texts
text_data = word_index_map(texts,word_index)
def test_pred(test_data,test):
    model = load_model('model_0218.h5')
    y_pred = model.predict(test_data)
    '''def gailv(y_pred,val_):
        
        gailv_0 = []
        gailv_1 = []
        gailv_2 = []
        for i in y_pred:
            gailv_0.append(i[0])
            gailv_1.append(i[1])
            gailv_2.append(i[2])
        val_['gailv_0'] = gailv_0
        val_['gailv_1'] = gailv_1
        val_['gailv_2'] = gailv_2
    gailv(y_pred,test)'''
    y_label = np.argmax(y_pred, axis=1)
    test['y_pred'] = y_label
    test.to_csv('test_pred.csv',index = False)
test_pred(test_data,test)
print('..............finished..............')

