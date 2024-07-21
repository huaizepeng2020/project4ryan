__author__="Michael Huai"
import numpy as np
import pandas as pd
import random
import os
import scipy.stats
import torch
import datetime
from datetime import timedelta
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import torch.nn as nn
from collections import OrderedDict
import math

class models_mh():
    def __init__(self,method_c,feature_name):
        if method_c==1:
            self.model=model1_OLS(feature_name)
        elif method_c==2:
            self.model=model2_lightgbm(feature_name)
        elif method_c==3:
            self.model=model3_mlp(feature_name)
        elif method_c==4:
            self.model=model4_FM(feature_name)
        elif method_c==5:
            self.model=model5_GRU(feature_name)
        elif method_c==6:
            self.model=model6_Transformer(feature_name)
        else:
            raise Exception("Invalid input model type")

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        self.model.train(data_x_train,data_y_train,data_x_valid,data_y_valid)

    def predict(self,data_x):
        predicts = self.model.predict(data_x)
        return predicts

class model1_OLS():
    def __init__(self,feature_name):
        self.name="OLS"
        self.time_falg = False
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.model = LinearRegression()


    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):

        all_datas_x=np.concatenate([data_x_train,data_x_valid],axis=0)
        all_datas_y=np.concatenate([data_y_train,data_y_valid],axis=0)
        # all_datas_x = sm.add_constant(all_datas_x)
        self.model.fit(all_datas_x, all_datas_y)

    def predict(self,data_x):
        predicts = self.model.predict(data_x)
        return predicts

class model2_lightgbm():
    def __init__(self,feature_name):
        self.name="Lightgbm"
        self.time_falg = False
        self.feature_name=feature_name
        self.num_feature=len(feature_name)

        self.params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 32,
            'learning_rate': 0.001,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 5,
            'verbose': -1,
            'tree_learner': 'data',
            "lambda_l2": 0.001,
        }

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        train_data = lgb.Dataset(data_x_train, label=data_y_train, free_raw_data=False)
        valid_data = lgb.Dataset(data_x_valid, label=data_y_valid, free_raw_data=False)
        callback = [lgb.early_stopping(stopping_rounds=10, verbose=False),
                    lgb.log_evaluation(period=2000, show_stdv=False)]
        self.model = lgb.train(self.params, train_set=train_data, valid_sets=valid_data, callbacks=callback,
                            num_boost_round=30000)

    def predict(self,data_x):
        predicts = self.model.predict(data_x)
        return predicts


class model3_mlp(nn.Module):
    def __init__(self,feature_name):
        super(model3_mlp, self).__init__()
        self.name="MLP"
        self.time_falg = False
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")

        self.model1 = MLP(self.num_feature, [1],dropout=0.5, batchnorm=True, activation='relu').to(self.device)
        self.model = MLP(self.num_feature, [1],dropout=0.5, batchnorm=True, activation='relu').to(self.device)
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def loss(self,pred_c,groundtruth_c):
        loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5
        # loss_c = torch.mean(torch.abs(pred_c-groundtruth_c))
        # loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5+torch.norm(pred_c)*1e-3
        return loss_c

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device)
        data_y_train=torch.tensor(data_y_train).to(torch.float32).to(self.device)
        data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device)
        data_y_valid=torch.tensor(data_y_valid).to(torch.float32).to(self.device)

        # data_x_train=torch.sign(data_x_train)*torch.log(torch.abs(data_x_train)+1)
        # data_x_valid=torch.sign(data_x_valid)*torch.log(torch.abs(data_x_valid)+1)

        best_epoch=0
        best_metric=1e9
        iner=0
        for epoch_c in range(1000000):
            self.model1.train()

            data_y_train_pred = self.model1(data_x_train)
            batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer1.zero_grad()
            batch_loss.backward()
            self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred = self.model1(data_x_valid)

                valid_metric=torch.mean(torch.abs(data_y_valid-data_y_valid_pred))

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                else:
                    iner += 1
                if iner>=10:
                    break

        for epoch_c in range(best_epoch+1):
            self.model.train()

            data_y_train_pred = self.model(data_x_train)
            data_y_valid_pred = self.model(data_x_valid)
            batch_loss = self.loss(torch.cat([data_y_train_pred,data_y_valid_pred],dim=0),torch.cat([data_y_train,data_y_valid],dim=0))
            # batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device)
        # data_x=torch.sign(data_x)*torch.log(torch.abs(data_x)+1)
        predicts = self.model(data_x).detach().numpy()
        return predicts


class model4_FM(nn.Module):
    def __init__(self,feature_name):
        super(model4_FM, self).__init__()
        self.name="FM"
        self.time_falg = False
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")

        self.model1 = MLP(int(self.num_feature*(1+self.num_feature)), [1],dropout=0.9, batchnorm=True, activation='relu').to(self.device)
        self.model = MLP(int(self.num_feature*(1+self.num_feature)), [1],dropout=0.9, batchnorm=True, activation='relu').to(self.device)
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def loss(self,pred_c,groundtruth_c):
        loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5
        # loss_c = torch.mean(torch.abs(pred_c-groundtruth_c))
        # loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5+torch.norm(pred_c)*1e-3
        return loss_c

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device)
        data_y_train=torch.tensor(data_y_train).to(torch.float32).to(self.device)
        data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device)
        data_y_valid=torch.tensor(data_y_valid).to(torch.float32).to(self.device)

        data_x_train=torch.cat([data_x_train,torch.multiply(data_x_train.unsqueeze(-1),data_x_train.unsqueeze(1)).reshape(len(data_x_train.unsqueeze(-1)),-1)],dim=1)
        data_x_valid=torch.cat([data_x_valid,torch.multiply(data_x_valid.unsqueeze(-1),data_x_valid.unsqueeze(1)).reshape(len(data_x_valid.unsqueeze(-1)),-1)],dim=1)

        # data_x_train=torch.sign(data_x_train)*torch.log(torch.abs(data_x_train)+1)
        # data_x_valid=torch.sign(data_x_valid)*torch.log(torch.abs(data_x_valid)+1)

        best_epoch=0
        best_metric=1e9
        iner=0
        for epoch_c in range(1000000):
            self.model1.train()

            data_y_train_pred = self.model1(data_x_train)
            batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer1.zero_grad()
            batch_loss.backward()
            self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred = self.model1(data_x_valid)

                valid_metric=torch.mean(torch.abs(data_y_valid-data_y_valid_pred))

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                else:
                    iner += 1
                if iner>=10:
                    break

        for epoch_c in range(best_epoch+1):
            self.model.train()

            data_y_train_pred = self.model(data_x_train)
            data_y_valid_pred = self.model(data_x_valid)
            batch_loss = self.loss(torch.cat([data_y_train_pred,data_y_valid_pred],dim=0),torch.cat([data_y_train,data_y_valid],dim=0))
            # batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device)
        data_x = torch.cat([data_x,torch.multiply(data_x.unsqueeze(-1), data_x.unsqueeze(1)).reshape(len(data_x.unsqueeze(-1)), -1)], dim=1)
        # data_x=torch.sign(data_x)*torch.log(torch.abs(data_x)+1)
        predicts = self.model(data_x).detach().numpy()
        return predicts


class model5_GRU(nn.Module):
    def __init__(self,feature_name):
        super(model5_GRU, self).__init__()
        self.name="GRU"
        self.time_falg=True
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")

        self.model1 = GRU_MLP(self.num_feature).to(self.device)
        self.model = GRU_MLP(self.num_feature).to(self.device)
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def loss(self,pred_c,groundtruth_c):
        loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5
        # loss_c = torch.mean(torch.abs(pred_c-groundtruth_c))
        # loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5+torch.norm(pred_c)*1e-3
        return loss_c

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_train=torch.tensor(data_y_train).to(torch.float32).to(self.device)
        data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_valid=torch.tensor(data_y_valid).to(torch.float32).to(self.device)

        best_epoch=0
        best_metric=1e9
        iner=0
        for epoch_c in range(1000000):
            self.model1.train()

            data_y_train_pred = self.model1(data_x_train)
            batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer1.zero_grad()
            batch_loss.backward()
            self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred = self.model1(data_x_valid)

                valid_metric=torch.mean(torch.abs(data_y_valid-data_y_valid_pred))

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                else:
                    iner += 1
                if iner>=10:
                    break

        for epoch_c in range(best_epoch+1):
            self.model.train()

            data_y_train_pred = self.model(data_x_train)
            data_y_valid_pred = self.model(data_x_valid)
            batch_loss = self.loss(torch.cat([data_y_train_pred,data_y_valid_pred],dim=0),torch.cat([data_y_train,data_y_valid],dim=0))
            # batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device).unsqueeze(1)
        predicts = self.model(data_x).detach().numpy()
        return predicts


class model6_Transformer(nn.Module):
    def __init__(self,feature_name):
        super(model6_Transformer, self).__init__()
        self.name = "Transformer"
        self.time_falg=True
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")


        self.model1 = Base_Transformer(max_len=5,input_size=self.num_feature).to(self.device)
        self.model = Base_Transformer(max_len=5,input_size=self.num_feature).to(self.device)
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def loss(self,pred_c,groundtruth_c):
        loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5
        # loss_c = torch.mean(torch.abs(pred_c-groundtruth_c))
        # loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5+torch.norm(pred_c)*1e-3
        return loss_c

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid):
        data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device)
        # data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_train=torch.tensor(data_y_train).to(torch.float32).to(self.device)
        data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device)
        # data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_valid=torch.tensor(data_y_valid).to(torch.float32).to(self.device)

        best_epoch=0
        best_metric=1e9
        iner=0
        for epoch_c in range(1000000):
            self.model1.train()

            data_y_train_pred = self.model1(data_x_train)
            batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer1.zero_grad()
            batch_loss.backward()
            self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred = self.model1(data_x_valid)

                valid_metric=torch.mean(torch.abs(data_y_valid-data_y_valid_pred))

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                else:
                    iner += 1
                if iner>=10:
                    break

        for epoch_c in range(best_epoch+1):
            self.model.train()

            data_y_train_pred = self.model(data_x_train)
            data_y_valid_pred = self.model(data_x_valid)
            batch_loss = self.loss(torch.cat([data_y_train_pred,data_y_valid_pred],dim=0),torch.cat([data_y_train,data_y_valid],dim=0))
            # batch_loss = self.loss(data_y_train_pred,data_y_train)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device).unsqueeze(0)
        predicts = self.model(data_x).detach().numpy()
        return predicts



class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers,
                 dropout=0.0, batchnorm=True, activation='relu'):
        super(MLP, self).__init__()
        modules = OrderedDict()

        previous_size = input_size
        for index, hidden_layer in enumerate(hidden_layers):
            if dropout and index!=len(hidden_layers)-1:
                modules[f"dropout{index}"] = nn.Dropout(dropout)
            modules[f"dense{index}"] = nn.Linear(previous_size, hidden_layer,bias=True)
            if batchnorm and index!=len(hidden_layers)-1:
                modules[f"batchnorm{index}"] = nn.BatchNorm1d(hidden_layer)
            if activation and index!=len(hidden_layers)-1:
                if activation.lower() == 'relu':
                    modules[f"activation{index}"] = nn.ReLU()
                elif activation.lower() == 'prelu':
                    modules[f"activation{index}"] = nn.PReLU()
                elif activation.lower() == 'sigmoid':
                    modules[f"activation{index}"] = nn.Sigmoid()
                else:
                    raise NotImplementedError(f"{activation} is not supported")
            previous_size = hidden_layer
        self._sequential = nn.Sequential(modules)

    def forward(self, input):
        return self._sequential(input)


class GRU_MLP(nn.Module):
    def __init__(self, input_size):
        super(GRU_MLP, self).__init__()
        modules = OrderedDict()

        self.gru = nn.GRU(input_size, 4, 1)
        self.mlp = nn.Linear(4, 1)
        self._sequential = nn.Sequential(modules)

    def forward(self, input):
        e1=self.gru(input)[1].squeeze(0)
        e2=self.mlp(e1)


        return e2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size,seq_len,dim = x.size()
        return self.encoding[:seq_len, :]

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 4. multiply with Value
        v = score @ v
        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    def forward(self, x):
        pos_emb = self.pos_emb(x)
        return self.drop_out(pos_emb.unsqueeze(0)+x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, max_len, d_model, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        fshape=x.shape
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Base_Transformer(nn.Module):
    def __init__(self, max_len,input_size):
        super(Base_Transformer, self).__init__()
        modules = OrderedDict()

        self.initial = nn.Linear(input_size, 4)

        self.m1 = Encoder(max_len,4,n_head=1,n_layers=1,drop_prob=0,device=torch.device("cpu"))
        self.mlp = nn.Linear(int(input_size*4), 1)
        self._sequential = nn.Sequential(modules)

    def forward(self, input):
        if input.dim==2:
            input=input.unsqueeze(0)
        x=self.initial(input)
        e1=self.m1(x)
        e2=self.mlp(e1.reshape(e1.shape[0],-1))

        return e2

