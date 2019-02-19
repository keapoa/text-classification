
# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model,load_model
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
import tensorflow as tf
import gc
import json
import jieba
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
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
        print('the macro f1 of validation data is {:.4f}'.format(_val_f1))
        if _val_f1 > self.best_f1:
            self.best_f1 = _val_f1
            self.model.save('model_0218.h5')
        return
train = pd.read_csv('train1_0215.csv')
valid = pd.read_csv('valid1_0215.csv')

words = [train,valid]

#分词
def token_word(words):
    result = []
    for word in words:
        out = [' '.join(jieba.cut(str(i))).split() for i in word['reviewContent']]
        result.append(out)
    return result
print('Start tokening...')
token_words = token_word(words)
texts = token_words[0]+token_words[1]

word_dict = json.load(open('wv.json','r'))


def sentences_encode_embedding(texts,maxseq = None,max_words = None):
    tokenizer = Tokenizer(num_words = max_words)
    tokenizer.fit_on_texts(texts)
    data_seq = tokenizer.texts_to_sequences(texts)
    data_pad = pad_sequences(data_seq, maxlen=maxseq)
    word_index = tokenizer.word_index
    word_len = len(word_index)
    #将词索引保存预测可能用
    '''def save_word_index(word_index):
        word_index['unk']=0  #加入未登录词
        with open('wi.json', 'w') as f:
        json.dump(word_index, f)
        f.close()
        return  '''
    embedding_word2vec = np.zeros((word_len + 1, 300))
    for word, i in word_index.items():
        embedding_vector = word_dict[word] if word in word_dict else None
        if embedding_vector is not None:
            embedding_word2vec[i] = embedding_vector
    train_data = data_pad[:train.shape[0]]
    val_data = data_pad[train.shape[0]:]
    return train_data, val_data,embedding_word2vec
print('Generate training text data and embedding vectors...')
train_data,val_data,embedding_word2vec = sentences_encode_embedding(texts,maxseq = 128,max_words = 50000)
def bi_gru_model(maxseq = None, embedding_dim = None,embedding_word2vec = None,gru_unit = None,num_classes = None ):
    content = Input(shape=(maxseq,), dtype='int32')
    embedding = Embedding(embedding_word2vec.shape[0], 
                          embedding_dim, 
                          weights=[embedding_word2vec],
                          trainable=False,
                          input_length=maxseq)

    x = SpatialDropout1D(0.2)(embedding(content))
    x0 = Bidirectional(CuDNNGRU(gru_unit, return_sequences=True))(x)
    x1 = Bidirectional(CuDNNGRU(gru_unit, return_sequences=True))(x0)
    
    avg_pool = GlobalAveragePooling1D()(x1)
    max_pool = GlobalMaxPooling1D()(x1)
    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.3)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=content, outputs=output)
    return model
val_ = valid.copy()

def train_gru(train_x=train_data, val_x = val_data):

    F1_score = 0
 
    gc.collect()
    K.clear_session()
    model = bi_gru_model(maxseq = 128,embedding_word2vec = embedding_word2vec,embedding_dim = 300,gru_unit = 200,num_classes = 3)
    model = multi_gpu_model(model,2)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
    train_y = train['y']
    val_y = valid['y']

    #test_y = test['y']
    y_train_onehot = to_categorical(train_y)
    y_val_onehot = to_categorical(val_y)

    history = model.fit([train_x], 
                  [y_train_onehot],
                  epochs=100,
                  batch_size=64, 
                  validation_data=(val_x, y_val_onehot),
                  callbacks = [ Metrics(),
                      EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='max'),
                           ReduceLROnPlateau(monitor="val_acc", verbose=0, mode='max', factor=0.5, patience=1)])
                  #ModelCheckpoint('bi_gru'+ '.hdf5', monitor='val_acc', verbose=0,save_best_only=True,mode='max',save_weights_only=True)])
    
    #model = load_model('bi_gru.h5',custom_objects={'AttentionWeightedAverage': AttentionWeightedAverage})'''
    model = load_model('model_0218.h5')
    print(model.summary()) 
    # 预测验证集和测试集
    y_val_pred = model.predict(val_x)

    #y_test_pred = model.predict(test_x)
    
    def gailv(y_pred,val_):
        
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

   
    y_val_pred = np.argmax(y_val_pred, axis=1)

    #y_test_pred = np.argmax(y_test_pred, axis=1)
    
    y_val_pred = y_val_pred.tolist()
    #y_test_pred = y_test_pred.tolist()
    
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
    print('the precision_0 accuarcy of validation data is {:.4f}'.format(precision_0))
    print('the precision_1 accuarcy of validation data is{:.4f}'.format(precision_1))
    print('the precision_2 accuarcy of validation data is{:.4f}'.format(precision_2))
    
    print('the reacll_0 of validation data is{:.4f}'.format(recall_0))
    print('the reacll_1 of validation data is{:.4f}'.format(recall_1))
    print('the reacll_2 of validation data is{:.4f}'.format(recall_2))
    
    print('the f1_0 of validation data is{:.4f}'.format(f1_0))
    print('the f1_1 of validation data is{:.4f}'.format(f1_1))
    print('the f1_2 of validation data is{:.4f}'.format(f1_2))
    
    print('the best macro f1 of validation data is {:.4f}'.format(f1))
    val_acc = accuracy_score(y_val_pred,val_y)
    print('the accuarcy of validation data is {:.4f}'.format(val_acc))
    val_f1 = f1_score(y_val_pred, val_y, average='macro')

    #test_f1 = f1_score(y_test_pred, test_y, average='macro')
    #print('the best macro f1 of test data is {}'.format(test_f1))
    val_['y_pred'] = y_val_pred

    #test_['y_pred'] = y_test_pred
    return 

print('Start train...')
out = train_gru(train_x=train_data, val_x = val_data)

#out.to_csv('bi_gru_shuju1.csv',index = False)
print('***************************finished****************************')

