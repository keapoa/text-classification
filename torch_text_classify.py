# coding: utf-8
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.autograd as autograd
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from tqdm import tqdm
import jieba
import json
train = pd.read_csv('train1_0215.csv')
valid = pd.read_csv('valid1_0215.csv')

def token_word(word):
    result = [' '.join(jieba.cut(str(i))).split() for i in word['reviewContent']]
    return result
train_fenci = token_word(train)
valid_fenci = token_word(valid)
texts = train_fenci+valid_fenci
#词频统计
def build_vocabulary(texts):
    vocabulary = []
    for text in texts:       
        for i in text:
            if i not in vocabulary:
                vocabulary.append(i)
    return vocabulary    
def word_frequency(texts):
    word = {}
    for text in texts:       
        for i in text:
            if i not in word:
                word[i] = 1
            else:
                word[i]+=1
    return word
def word_index(texts):
    word_idx = {}
    for idx,word in enumerate(word_frequency(texts).keys()):
        word_idx[word] = idx+1
    return word_idx
     
def encode_samples(tokenized_samples,word_index):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_index:
                feature.append(word_index[token])
            else:
                feature.append(0)
        features.append(feature)
    return features
def pad_samples(features, maxlen=128):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(0)
        padded_features.append(padded_feature)
    return padded_features
def weights(vocabulary,word_to_index):
    word_dict = json.load(open('wv.json','r'))
    embedding_word2vec = torch.zeros((len(vocabulary) + 1, 300))
    for word, i in word_to_index.items():
        embedding_vector = word_dict[word] if word in word_dict else None
        if embedding_vector is not None:
            embedding_word2vec[i] = torch.Tensor(embedding_vector)
    return embedding_word2vec

word_to_index = word_index(texts)
train_encode = pad_samples(encode_samples(train_fenci,word_to_index))
valid_encode = pad_samples(encode_samples(valid_fenci,word_to_index))
vocabulary = build_vocabulary(texts)
weight = weights(vocabulary,word_to_index) 

print(np.array(train_encode).shape)
#生成数据集
train_features = torch.tensor(train_encode)
train_labels = torch.tensor(train['y'].values)
valid_features = torch.tensor(valid_encode)
valid_labels = torch.tensor(valid['y'].values)

#超参数
num_epochs = 5
hidden_size = 200
batch_size = 64
num_layers = 1
time_step = 128
embed_dim = 300
lr = 0.001
device = torch.device('cuda:0')
use_gpu = True


#构建迭代数据集
from torch.utils.data import TensorDataset,DataLoader
train_data = TensorDataset(train_features, train_labels)
valid_data = TensorDataset(valid_features, valid_labels)

train_iter = DataLoader(train_data, batch_size=batch_size,shuffle=True)
valid_iter = DataLoader(valid_data, batch_size=batch_size,shuffle=False)

#构建网络

class text_gru(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes,
                 weight,bidirectional,use_gpu):
        super(text_gru,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional=True,
            dropout = 0
        )
        self.MaxPool = nn.MaxPool1d(kernel_size=128)
        if self.bidirectional:
            self.linear = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.linear = nn.Linear(hidden_size * 1, num_classes)

    def forward(self,inputs):
        embed_out = self.embedding(inputs)
        out, hidden = self.gru(embed_out)#the requred dimension of input is [batch_size,time_step,hidden_size] batch_first = True
        output = out.permute([0,2,1])#the dimension of output is [batch_size,hidden_size,time_step]

        output = self.MaxPool(output)

        output = self.linear(output[:,:,0])

        return output

model = text_gru(input_size=300,hidden_size=200,num_layers=1,num_classes=3,
              weight=weight,bidirectional=True,use_gpu=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#训练
valid_best_f1 = 0
for epoch in range(num_epochs):
    train_loss, valid_losses = 0, 0
    train_acc, valid_acc = 0, 0
    n,m = 0,0
    f1 = 0
    y_train_pred = torch.cuda.LongTensor([])
    y_train_true = torch.cuda.LongTensor([])
    for feature, label in train_iter:
        n += 1
        model.zero_grad()
        feature = Variable(feature.cuda())
        label = Variable(label.cuda())
        score = model(feature)
        loss = criterion(score, label)
        loss.backward()
        optimizer.step()
        y_train_pred = torch.cat([y_train_pred,torch.argmax(score,dim=1)])
        y_train_true = torch.cat([y_train_true,label])
        #train_acc += float(accuracy_score(torch.argmax(score,dim=1), label))
        train_loss += float(loss)
    with torch.no_grad():
        y_valid_pred = torch.cuda.LongTensor([])
        y_valid_true = torch.cuda.LongTensor([])
        for valid_feature, valid_label in valid_iter:
            m+=1
            valid_feature = valid_feature.cuda()
            valid_label = valid_label.cuda()
            valid_score = model(valid_feature)
            valid_loss = criterion(valid_score, valid_label)
            y_pred = torch.argmax(valid_score,dim=1)
            y_valid_pred = torch.cat([y_valid_pred,y_pred])
            y_valid_true = torch.cat([y_valid_true,valid_label])
            valid_losses += float(valid_loss)
    train_acc = accuracy_score(y_train_pred, y_train_true)
    valid_acc = accuracy_score(y_valid_pred, y_valid_true)
    valid_f1 = f1_score(y_valid_pred, y_valid_true,average='macro')
    print('epoch: {}, train loss: {:.4f}, train acc: {:.3f}, valid loss: {:.4f}, valid acc: {:.3f}, f1:{:.3f}'.format(epoch, train_loss / n, train_acc, valid_losses / m, valid_acc, valid_f1))
    if valid_best_f1<=valid_f1:
        print('当前保存第{}次迭代'.format(epoch))
        valid_best_f1 = valid_f1
        torch.save(model.state_dict(), 'model.ckpt')
'''#torch.save(model.state_dict(), 'model.pkl')
model.load_state_dict(torch.load('model.pkl'))
m=0
valid_losses = 0
valid_acc = 0
for valid_feature, valid_label in valid_iter:    
    m += 1
    valid_score = model(valid_feature.cuda())
    valid_loss = criterion(valid_score, valid_label.cuda())#loss可导变量会保存在内存中
    valid_acc += accuracy_score(torch.argmax(valid_score,
                                                    dim=1), valid_label)
    valid_losses += float(valid_loss)#缓解cuda，内存超出情况
print('valid loss: {:.4f}, valid acc: {:.2f}'.format(valid_losses / m, valid_acc / m))'''
