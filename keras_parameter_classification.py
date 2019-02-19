#！/root/anaconda3/bin/python3.6
import pandas as pd
import numpy as np
import random
import jieba 
from gensim.models import word2vec 
import warnings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model,load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
import tensorflow as tf
import gc
import re
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning,module = 'gensim' )



flags = tf.flags

FLAGS = flags.FLAGS

#定义参数
#Required parameters
flags.DEFINE_string("train_file_path",None,"train_file_path")
flags.DEFINE_string("valid_file_path",None,"valid_file_path")
flags.DEFINE_string("test_file_path",None,"valid_file_path")
flags.DEFINE_string("data_dir", None,"The data file dir")
#model parameters
flags.DEFINE_integer("max_seq_length",128,"The maximum total input sequence length after tokenization. ")
flags.DEFINE_integer("units",200,"hidden units")
flags.DEFINE_float("SpatialDropout",0.2,"GRU Dropout")
flags.DEFINE_float("dropout",0.3,"Dropout probability")
flags.DEFINE_integer("num_classes",3,"the nums of the classification")
#word2vec parameters
flags.DEFINE_integer("embedding_dim",300,"Dimension of word embedding")
flags.DEFINE_integer("min_count",2,"The smallest word frequency")
flags.DEFINE_integer("window",6,"The window size")
#training parameters
flags.DEFINE_integer("max_words",5000,"Keep the counts of the word accoroding to frequency ")
flags.DEFINE_integer("batch_size",64,"Batch_size")
flags.DEFINE_integer("num_epoch",100,"Number of training epochs")
#bool parameters
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_valid", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_test", False, "Whether to run testing.")
#Gpu Parameter
flags.DEFINE_integer("num_gpu",1,'the count of gpu')
#export file
flags.DEFINE_string("output_dir", None,"The output directory where the result of the prediction.")
flags.DEFINE_bool("valid_out", False, "Whether to export valid file.")
flags.DEFINE_bool("test_out", False, "Whether to export test file.")

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
            self.model.save('best_model.h5')
        return

class Dataprocessor(object):
    def __init__(self,data_dir,train_file_path,valid_file_path,test_file_path):
        self.train_file = pd.read_csv(os.path.join(data_dir,train_file_path))
        self.valid_file = pd.read_csv(os.path.join(data_dir,valid_file_path))
        self.test_file = pd.read_csv(os.path.join(data_dir,test_file_path))
    def token_word(self):
        words = [self.train_file,self.valid_file,self.test_file]
        result = []
        for word in words:
            out = [' '.join(jieba.cut(str(i))).split() for i in word['reviewContent']]
            result.append(out)
        return result
    def sentences_encode_embedding(self,maxseq = None,max_words = None,embedding_dim = None,min_count = None,window = None):
        #设置最大长度为128
        token_words = self.token_word()
        texts = token_words[0]+token_words[1]+token_words[2]
        tokenizer = Tokenizer(num_words = max_words)
        tokenizer.fit_on_texts(texts)
        data_seq = tokenizer.texts_to_sequences(texts)
        data_pad = pad_sequences(data_seq, maxlen=maxseq)
    
        model_dm = word2vec.Word2Vec(texts,
                             size = embedding_dim,
                             sg= 1,
                             min_count = min_count,
                             window = window,
                             seed = 2018)
  
        count = 0
        word_index = tokenizer.word_index
        with open("word_dict.txt","w",encoding = "utf-8") as f:
            f.write(str(word_index))
        f.close()
        word_len = len(word_index)
        embedding_word2vec = np.zeros((word_len + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = model_dm[word] if word in model_dm else None
            if embedding_vector is not None:
                count += 1
                embedding_word2vec[i] = embedding_vector
            else:
                unk_vec = np.random.random(embedding_dim) * 0.5
                unk_vec = unk_vec - unk_vec.mean()
                embedding_word2vec[i] = unk_vec

        print('The counts of effective word:',count)  
        train_data = data_pad[:self.train_file.shape[0]]
        val_data = data_pad[self.train_file.shape[0]:(self.train_file.shape[0]+self.valid_file.shape[0])]
        test_data = data_pad[(self.train_file.shape[0]+self.valid_file.shape[0]):]
    
        return train_data, val_data,test_data, embedding_word2vec

class Model_train(object):
    def __init__(self,train_x,val_x,test_x,embedding_word2vec,train,valid,test):
        self.train_x = train_x
        self.val_x = val_x
        self.test_x = test_x
        self.embedding_word2vec = embedding_word2vec
        self.train = train
        self.valid = valid
        self.test = test
        
    def build_model(self):
        content = Input(shape=(FLAGS.max_seq_length,), dtype='int32')
        embedding = Embedding(self.embedding_word2vec.shape[0], 
                          FLAGS.embedding_dim, 
                          weights=[self.embedding_word2vec],
                          trainable=False,
                          input_length=FLAGS.max_seq_length)

        x = SpatialDropout1D(FLAGS.SpatialDropout)(embedding(content))
        x0 = Bidirectional(CuDNNGRU(FLAGS.units, return_sequences=True))(x)
        x1 = Bidirectional(CuDNNGRU(FLAGS.units, return_sequences=True))(x0)
    
        avg_pool = GlobalAveragePooling1D()(x1)
        max_pool = GlobalMaxPooling1D()(x1)
        conc = concatenate([avg_pool, max_pool])

        x = Dropout(FLAGS.dropout)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
        x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
        output = Dense(FLAGS.num_classes, activation="softmax")(x)

        model = Model(inputs=content, outputs=output)
        return model
    def train_model(self):
        F1_score = 0
        gc.collect()
        K.clear_session()
        model = self.build_model()
        if FLAGS.num_gpu>1:
            model = multi_gpu_model(model,FLAGS.num_gpu)
        model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
        train_y = self.train['y']
        val_y = self.valid['y']
        test_y = self.test['y']
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        history = model.fit([self.train_x], 
                  [y_train_onehot],
                  epochs=FLAGS.num_epoch,
                  batch_size=FLAGS.batch_size, 
                  validation_data=(self.val_x, y_val_onehot),
                  callbacks = [ Metrics(),
                           EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='max'),
                           ReduceLROnPlateau(monitor="val_acc", verbose=0, mode='max', factor=0.5, patience=1)])
                  #ModelCheckpoint('bi_gru'+ '.hdf5', monitor='val_acc', verbose=0,save_best_only=True,mode='max',save_weights_only=True)])
        model = load_model('best_model.h5')
        # 预测验证集和测试集
        y_val_pred = model.predict(self.val_x)
        y_test_pred = model.predict(self.test_x)

            
        y_val_pred = np.argmax(y_val_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)
    
        y_val_pred = y_val_pred.tolist()
        y_test_pred = y_test_pred.tolist()
    
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
        test_f1 = f1_score(y_test_pred, test_y, average='macro')
        print('the best macro f1 of test data is {}'.format(test_f1))
        if FLAGS.valid_out:
            val_ = dataprocess.valid_file.copy()
            val_['y_pred'] = y_val_pred
            output_val_file = os.path.join(FLAGS.output_dir,'val_out.csv')
            val_.to_csv(output_val_file,index = False)
        if FLAGS.test_out:
            test_ = dataprocess.test_file.copy()
            test_['y_pred'] = y_test_pred
            output_test_file = os.path.join(FLAGS.output_dir, "test_out.csv")
            test_.to_csv(output_test_file,index = False)
        return 
class Model_test(object):
    def __init__(self,data_dir,test_file_path):
        self.test_file = pd.read_csv(os.path.join(data_dir,test_file_path))
    def build_test_seq(self):
        token_words = [' '.join(jieba.cut(str(i))).split() for i in self.test_file['reviewContent']]
        test_pad = []
        with open('word_dict.txt','r',encoding = 'utf-8') as f:
            word_dict = eval(f.read())
        f.close()
        for sentence in token_words:
            one_sentence = []
            for j in sentence:
                if j in word_dict.keys():
                    one_sentence.append(word_dict[j])
                else:
                    one_sentence.append(0)
            if len(sentence) <= FLAGS.max_seq_length:
                one_sentence = [0]*(FLAGS.max_seq_length-len(sentence))+one_sentence
            else:
                one_sentence = one_sentence[(len(sentence)-FLAGS.max_seq_length):]
            test_pad.append(one_sentence)
        return np.array(test_pad)
    def test_pre(self):
        model = load_model('best_model.h5')
        y_test_pred = model.predict(self.build_test_seq())
        y_test_pred = np.argmax(y_test_pred, axis=1)
        test_f1 = f1_score(y_test_pred, self.test_file['y'], average='macro')
        print('the macro f1 of the test data is {}'.format(test_f1))
        return 
       
    
def main(_):
    if FLAGS.do_train:
        dataprocess = Dataprocessor(FLAGS.data_dir, 
                           FLAGS.train_file_path, 
                           FLAGS.valid_file_path,
                           FLAGS.test_file_path)
        #dataprocess = Dataprocessor('',"train_review.csv","valid_review.csv","test_review.csv")
        train_x,val_x,test_x,embedding_word2vec = dataprocess.sentences_encode_embedding(maxseq = FLAGS.max_seq_length,
                                                              max_words = FLAGS.max_words,
                                                              embedding_dim = FLAGS.embedding_dim,
                                                              min_count = FLAGS.min_count,
                                                              window = FLAGS.window)
        model = Model_train(train_x,
                      val_x,
                      test_x,
                      embedding_word2vec,
                      dataprocess.train_file,
                      dataprocess.valid_file,
                      dataprocess.test_file)
        print('Start train...')
        model.train_model()
    if FLAGS.do_test:
        model = Model_test(FLAGS.data_dir, 
                     FLAGS.test_file_path)
        model.test_pre()
if __name__ == "__main__":
    tf.app.run()
print('***************************finished****************************')

