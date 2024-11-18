__author__="Michael Huai"
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import torch.nn as nn
from collections import OrderedDict
import math
from base_model2 import Crossformer
from torch.utils.data import DataLoader,Dataset
import multiprocessing
num_cores=int(0.9*multiprocessing.cpu_count())
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
from models.model.transformer import Transformer

class Dataset_MH(Dataset):
    def __init__(self, data_x, data_y,data_y_reg):
        # size [seq_len, label_len, pred_len]
        # info
        self.data_x = data_x
        self.data_y = data_y
        self.data_y_reg = data_y_reg

    def __getitem__(self, index):

        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_y_reg = self.data_y_reg[index]

        return seq_x, seq_y, seq_y_reg

    def __len__(self):
        return len(self.data_x)

class models_mh():
    def __init__(self,method_c,feature_name,config_c=None):
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
            self.model=model6_MHformer(feature_name,config_c)
        elif method_c==7:
            self.model=model7_Transformer(feature_name,config_c)
        else:
            raise Exception("Invalid input model type")

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid,add=None):
        if self.model.name=="MHsformer":
            self.model.train(data_x_train,data_y_train,data_x_valid,data_y_valid,add)
        else:
            self.model.train(data_x_train, data_y_train, data_x_valid, data_y_valid)

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


class model6_MHformer(nn.Module):
    def __init__(self,feature_name,config_c):
        super(model6_MHformer, self).__init__()
        self.name = "MHsformer"
        self.time_falg=True
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.len_t=config_c["in_len"]
        data_dim=len(feature_name)
        in_len=config_c["in_len"]
        self.in_len=in_len
        out_len=config_c["out_len"]
        self.out_len = out_len
        d_model=config_c["d_model"]
        self.d_model = d_model
        d_ff=config_c["d_ff"]
        self.d_ff = d_ff
        n_heads=config_c["n_heads"]
        self.n_heads = n_heads
        n_layers=config_c["n_layers"]
        self.n_layers = n_layers
        dropout=config_c["dropout"]
        self.dropout = dropout
        rate_loss=config_c["rate_loss"]
        self.rate_loss=1/np.array(rate_loss)
        self.rate_loss[0] = self.rate_loss[0]*0.5
        self.rate_loss[2] = self.rate_loss[2]*0.5
        batch_size = config_c["batch_size"]
        self.batch_size=batch_size

        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")


        # input_size,d_model, n_head, max_len,ffn_hidden, n_layers, drop_prob=0, device=torch.device("cpu")):
        self.model1 = Base_MHformer(
            input_size=self.num_feature,
            d_model=d_model,
            n_head=n_heads,
            max_len=in_len,
            ffn_hidden=d_ff,
            n_layers=n_layers,
            drop_prob=dropout,
            device=self.device,
            reg_input=False,
            reg_output=False)
        self.model = Base_MHformer(
            input_size=self.num_feature,
            d_model=d_model,
            n_head=n_heads,
            max_len=in_len,
            ffn_hidden=d_ff,
            n_layers=n_layers,
            drop_prob=dropout,
            device=self.device,
            reg_input=False,
            reg_output=False)
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.0001,weight_decay=0.0001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001,weight_decay=0.0001)

    def loss(self,pred_c,groundtruth_c,out_c1=None,out_c2=None,batch_y_reg=None):
        idx11=torch.where(groundtruth_c==0)[0]
        idx12=torch.where(groundtruth_c==1)[0]
        idx13=torch.where(groundtruth_c==2)[0]
        weight_c=torch.ones(len(groundtruth_c),3)
        weight_c[idx11]=self.rate_loss[0]
        weight_c[idx12]=self.rate_loss[1]
        weight_c[idx13]=self.rate_loss[2]
        loss_f = nn.BCELoss(weight=weight_c)
        groundtruth_c_one_hot = torch.sparse.torch.eye(3).index_select(0,groundtruth_c.int()).to(self.device)
        loss_c=loss_f(pred_c,groundtruth_c_one_hot)

        if out_c1!=None:
            batch_y_reg1=batch_y_reg.reshape(-1)
            out_c1=out_c1.reshape(-1,out_c1.shape[-1])
            idx11 = torch.where(batch_y_reg1 == 0)[0]
            idx12 = torch.where(batch_y_reg1 == 1)[0]
            idx13 = torch.where(batch_y_reg1 == 2)[0]
            weight_c = torch.ones(len(batch_y_reg1), 3)
            weight_c[idx11] = self.rate_loss[0]
            weight_c[idx12] = self.rate_loss[1]
            weight_c[idx13] = self.rate_loss[2]
            loss_f = nn.BCELoss(weight=weight_c)
            groundtruth_c_one_hot = torch.sparse.torch.eye(3).index_select(0, batch_y_reg1.int()).to(self.device)
            loss_c1 = loss_f(out_c1, groundtruth_c_one_hot)

            loss_c = loss_c + 0.1*loss_c1

        if out_c2!=None:
            batch_y_reg2=batch_y_reg[:,1:].reshape(-1)
            out_c2=out_c2[:,:-1].reshape(-1,out_c2.shape[-1])
            idx11 = torch.where(batch_y_reg2 == 0)[0]
            idx12 = torch.where(batch_y_reg2 == 1)[0]
            idx13 = torch.where(batch_y_reg2 == 2)[0]
            weight_c = torch.ones(len(batch_y_reg2), 3)
            weight_c[idx11] = self.rate_loss[0]
            weight_c[idx12] = self.rate_loss[1]
            weight_c[idx13] = self.rate_loss[2]
            loss_f = nn.BCELoss(weight=weight_c)
            groundtruth_c_one_hot = torch.sparse.torch.eye(3).index_select(0, batch_y_reg2.int()).to(self.device)
            loss_c2 = loss_f(out_c2, groundtruth_c_one_hot)

            loss_c = loss_c + 0.1*loss_c2

        return loss_c

    def train(self,data_x_train,data_y_train,data_x_valid,data_y_valid,add=None):
        data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device)
        # data_x_train=torch.tensor(data_x_train).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_train=torch.tensor(data_y_train).to(torch.float32).to(self.device)
        data_y_reg_train=torch.tensor(add).to(torch.float32).to(self.device)
        data_y_reg_train=data_y_reg_train.reshape(data_y_reg_train.shape[0],data_y_reg_train.shape[1])
        data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device)
        # data_x_valid=torch.tensor(data_x_valid).to(torch.float32).to(self.device).permute(1,0,2)
        data_y_valid=torch.tensor(data_y_valid).to(torch.float32).to(self.device)

        best_epoch=0
        best_metric=1e9
        iner=0
        train_loader = DataLoader(
            Dataset_MH(data_x_train, data_y_train,data_y_reg_train),
            batch_size=self.batch_size, shuffle=True)

        for epoch_c in range(1000000):
            self.model1.train()

            for i, (batch_x,batch_y,batch_y_reg) in enumerate(train_loader):
                data_y_train_pred,out_c1,out_c2 = self.model1(batch_x)
                batch_loss = self.loss(data_y_train_pred,batch_y,out_c1,out_c2,batch_y_reg)

                self.optimizer1.zero_grad()
                batch_loss.backward()
                self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred,_,_ = self.model1(data_x_valid)
                valid_metric = self.loss(data_y_valid_pred, data_y_valid)

                class_c=np.argmax(data_y_valid_pred,axis=1)
                from sklearn.metrics import classification_report
                t = classification_report(data_y_valid, class_c, target_names=['空', 'ping', '多'],output_dict=True)
                valid_metric=-(t["空"]["precision"]+t["多"]["precision"])
                print(f"----{epoch_c}---{valid_metric}----")

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                    iner=0
                else:
                    iner += 1
                if iner>=10:
                    break
        print(f"---best epoch is {best_epoch}-----")
        # train_loader = DataLoader(
        #     Dataset_MH(data_x_train, data_y_train,data_y_reg_train),
        #     batch_size=self.batch_size, shuffle=True)

        for epoch_c in range(best_epoch+1):
            self.model.train()

            for i, (batch_x, batch_y,batch_y_reg) in enumerate(train_loader):
                data_y_train_pred,out_c1,out_c2 = self.model(batch_x)
                batch_loss = self.loss(data_y_train_pred,batch_y,out_c1,out_c2,batch_y_reg)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

        with torch.no_grad():
            self.model.eval()
            data_y_valid_pred, _, _ = self.model(data_x_valid)
            class_c = np.argmax(data_y_valid_pred, axis=1)
            t = classification_report(data_y_valid, class_c, target_names=['空', 'ping', '多'], output_dict=True)
            valid_metric = -(t["空"]["precision"] + t["多"]["precision"])
            print(f"----valid---{epoch_c}---{valid_metric}----")

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device)
        predicts,_,_ = self.model(data_x)
        predicts = predicts.detach().numpy()
        predicts = np.argmax(predicts,axis=1)
        return predicts


class model7_Transformer(nn.Module):
    def __init__(self,feature_name,config_c):
        super(model7_Transformer, self).__init__()
        self.name = "MHformer"
        self.time_falg=True
        self.feature_name=feature_name
        self.num_feature=len(feature_name)
        self.device = torch.device("cuda:" + str(0)) if False else torch.device("cpu")

        self.model1 = Crossformer(feature_name,config_c).to(self.device) # data_dim, in_len, out_len, seg_len,
        self.model = Crossformer(feature_name,config_c).to(self.device) # data_dim, in_len, out_len, seg_len,
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def loss(self,pred_c,groundtruth_c):
        idx11=torch.where(groundtruth_c==0)[0]
        idx12=torch.where(groundtruth_c==1)[0]
        idx13=torch.where(groundtruth_c==2)[0]
        weight_c=torch.ones(len(groundtruth_c),3)
        weight_c[idx11]=len(idx12)/len(idx11)
        weight_c[idx13]=len(idx12)/len(idx13)
        loss_f = nn.BCELoss(weight=weight_c)
        groundtruth_c_one_hot = torch.sparse.torch.eye(3).index_select(0,groundtruth_c.int()).to(self.device)
        loss_c=loss_f(pred_c,groundtruth_c_one_hot)

        # loss_c = torch.mean((pred_c-groundtruth_c)**2)**0.5

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
        train_loader = DataLoader(
            Dataset_MH(data_x_train, data_y_train),
            batch_size=1024, shuffle=True)

        for epoch_c in range(1000000):
            self.model1.train()

            for i, (batch_x,batch_y) in enumerate(train_loader):
                data_y_train_pred = self.model1(batch_x)
                batch_loss = self.loss(data_y_train_pred,batch_y)

                self.optimizer1.zero_grad()
                batch_loss.backward()
                self.optimizer1.step()

            with torch.no_grad():
                self.model1.eval()
                data_y_valid_pred = self.model1(data_x_valid)
                valid_metric = self.loss(data_y_valid_pred, data_y_valid)

                if valid_metric<=best_metric:
                    best_epoch=epoch_c
                    best_metric=valid_metric
                else:
                    iner += 1
                if iner>=10:
                    break

        for epoch_c in range(best_epoch+1):
            self.model.train()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                data_y_train_pred = self.model(batch_x)
                batch_loss = self.loss(data_y_train_pred, batch_y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.model.eval()

    def predict(self,data_x):
        data_x=torch.tensor(data_x).to(torch.float32).to(self.device)
        predicts = self.model(data_x).detach().numpy()
        predicts = np.argmax(predicts,axis=1)
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

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

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
            score = score.masked_fill(mask == 0, -1000000000)
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

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):
    def __init__(self, max_len, d_model,ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.mask = torch.tril(torch.ones(max_len, max_len), diagonal=0)
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, self.mask)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.mask = torch.tril(torch.ones(max_len, max_len), diagonal=0)
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, self.mask, self.mask)
        # # pass to LM head
        # output = self.linear(trg)
        return trg

class Base_MHformer(nn.Module):
    def __init__(self,  input_size,d_model, n_head, max_len,ffn_hidden, n_layers, drop_prob=0, device=torch.device("cpu"),reg_input=False,reg_output=False):
        super(Base_MHformer, self).__init__()
        self.reg_input=reg_input
        self.reg_output=reg_output

        self.initial = nn.Sequential(nn.Linear(input_size, d_model))
        # self.norm1 = LayerNorm(d_model=d_model)

        if reg_input:
            # self.reg_input_nn = nn.Sequential(nn.Linear(d_model, 1),nn.Tanh())
            self.reg_input_nn = nn.Sequential(nn.Linear(d_model, 3), nn.Sigmoid())

        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        if reg_output:
            self.reg_output_nn = nn.Sequential(nn.Linear(d_model, 3),nn.Sigmoid())


        # self.mlp = nn.Linear(int(max_len * d_model), 3)
        # self.mlp = nn.Linear(int(d_model), 3)
        # self.mlp = nn.Sequential(nn.Linear(int(max_len * d_model), 3),nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(d_model, 3),nn.Sigmoid())

    def forward(self, src):
        src=self.initial(src)
        if self.reg_input:
            # out_c1=self.reg_input_nn(src).squeeze(-1)
            out_c1=self.reg_input_nn(src)
        else:
            out_c1=None
        enc_src = self.encoder(src)
        output = self.decoder(trg=enc_src,src=None)
        if self.reg_output:
            out_c2=self.mlp(output)
            # out_c2=self.reg_output_nn(output).squeeze(-1)
        else:
            out_c2=None
        e2=self.mlp(output[:,-1])
        # e2=self.mlp(output.reshape(output.shape[0],-1))
        return e2,out_c1,out_c2


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask


    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward1(self, input):
        if input.dim==2:
            input=input.unsqueeze(0)
        x=self.initial(input)
        e1=self.m1(x)
        e2=self.mlp(e1.reshape(e1.shape[0],-1))
        e2=torch.sigmoid(e2)

        return e2

