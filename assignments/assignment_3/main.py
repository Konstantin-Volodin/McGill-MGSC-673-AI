#%% IMPORTS
import pandas as pd
import numpy as np
import json
import itertools

import sklearn.preprocessing
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
# from lightning.pytorch.callbacks import ModelCheckpoint


##### DATASET DEFINITION #####
# https://www.kaggle.com/datasets/abhinav89/telecom-customer
class CustomerChurn(Dataset):
    """Customer Churn Dataset."""

    def __init__(self, data_file, dictionary_file, transform=None):
        self.df = pd.read_csv(data_file)
        with open(dictionary_file) as f: 
            self.df_dict = json.load(f)

        self.__get_col_types__()
        self.__transform_data__()

    
    def __get_col_types__(self):
        """Identifies Numerical, Categorical, and Target Columns"""
        self.num_cols = []
        self.cat_cols = []
        self.emb_cols = []
        self.targ = 'churn'

        for col in self.df.columns:
            type = self.df[col].dtype
            uniques = self.df.value_counts(col).shape[0]
            
            # Custom Logic
            if col == 'Customer_ID' or col == 'churn':
                continue

            # Other Cols
            if type in ['float64', 'int64']:
                self.num_cols.append(col)
            elif type in ['object'] and uniques > 10:
                self.emb_cols.append(col)
            elif type in ['object'] and uniques <= 10:
                self.cat_cols.append(col)

    def __transform_data__(self):
        num_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='median')),
                             ('transform', sklearn.preprocessing.StandardScaler()),])
        cat_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='most_frequent')),
                             ('transform', sklearn.preprocessing.OneHotEncoder(drop='first')),])
        
        ##### TODO: add entity embedding for emb_cols to reduce number of groups


        tr_pipe = ColumnTransformer([ ("numerical", num_pipe, self.num_cols),
                                      ("categorical", cat_pipe, self.cat_cols), ])
        df_x = tr_pipe.fit_transform(self.df)
        self.df_x = pd.DataFrame(df_x, columns=tr_pipe.get_feature_names_out())

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_vals = self.df_x.iloc[idx, ]
        y_vals = self.df[self.targ].iloc[idx, ]

        x_tensor = torch.Tensor(x_vals.values)
        y_tensor = torch.Tensor([y_vals]).T

        return (x_tensor, y_tensor)

##### NEURAL NETWORK DEFINITION #####
class DenseNN(pl.LightningModule):
    def __init__(self, hd, ls, dr, af, opt):
        super().__init__()
        encoder = nn.Sequential()
        self.optimizer_type = opt
    
        # Input Layer
        encoder.append(nn.Linear(108, ls))
        encoder.append(nn.Dropout(dr))
        if af == 'ReLU': encoder.append(nn.ReLU())
        elif af =='LeakyReLU': encoder.append(nn.LeakyReLU())
        elif af =='ELU': encoder.append(nn.ELU())
        elif af =='Tanh': encoder.append(nn.Tanh())

        # Hidden Layers
        for i in range(hd-1):  
            encoder.append(nn.Linear(ls, ls))
            encoder.append(nn.Dropout(dr))
            if af == 'ReLU': encoder.append(nn.ReLU())
            elif af =='LeakyReLU': encoder.append(nn.LeakyReLU())
            elif af =='ELU': encoder.append(nn.ELU())
            elif af =='Tanh': encoder.append(nn.Tanh())

        # Output Layers
        encoder.append(nn.Linear(ls, 1))
        encoder.append(nn.Dropout(dr))
        encoder.append(nn.Sigmoid())

        self.encoder = encoder

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        if self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        elif self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        elif self.optimizer_type == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y_hat = self.encoder(x)    
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y_hat = self.encoder(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)


##### DATA PREPARATION #####
data = CustomerChurn('data.csv', 'data_dict.json')
data_train, data_val, data_test = random_split(data, [40000, 10000, 50000])
data_train = DataLoader(data_train, batch_size=500)
data_val = DataLoader(data_val, batch_size=500)
data_test = DataLoader(data_test, batch_size=500)


#%%
##### BASELINE MODEL #####
hl_def = 1
ls_def = 256
dr_def = 0.25
act_def = 'ReLU'
opt_def = 'Adam'

logger = TensorBoardLogger('lightning_logs', name='baseline')
model = DenseNN(hl_def, ls_def, dr_def, act_def, opt_def)
trainer = pl.Trainer(max_epochs=50, logger=logger)
trainer.fit(model, data_train, data_val)

#%%
##### EXPERIMENTATION #####
# Optimizers
optimizers = ['Adam', 'SGD', 'LBFGS']
for opt in optimizers:
    if opt == opt_def: continue
    logger = TensorBoardLogger('lightning_logs', name=f'opt_{opt}')
    model = DenseNN(hl_def, ls_def, dr_def, act_def, opt)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, data_train, data_val)

#%%
# Network Architecture (layer size)
layer_sizes = [32, 64, 128, 256, 512]
for ls in layer_sizes:
    if ls == ls_def: continue
    logger = TensorBoardLogger('lightning_logs', name=f'ls_{ls}')
    model = DenseNN(hl_def, ls, dr_def, act_def, opt_def)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, data_train, data_val)


#%%
# Network Architecture (hidden layers)
hidden_layers = [1,2,3]
for hl in hidden_layers:
    if hl == hl_def: continue
    logger = TensorBoardLogger('lightning_logs', name=f'hl_{hl}')
    model = DenseNN(hl, ls_def, dr_def, act_def, opt_def)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, data_train, data_val)


#%%
# Network Architecture (Dropout Rate)
dropout_rates = np.linspace(0,0.9, 10)
for dr in dropout_rates:
    if dr == dr_def: continue
    logger = TensorBoardLogger('lightning_logs', name=f'dr_{dr}')
    model = DenseNN(hl_def, ls_def, dr, act_def, opt_def)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, data_train, data_val)


#%% 
# Activation Functions
activations = ['ReLU', 'LeakyReLU', 'ELU', 'Tanh']
for act in activations:
    if act == act_def: continue
    logger = TensorBoardLogger('lightning_logs', name=f'act_{act}')
    model = DenseNN(hl_def, ls_def, dr_def, act, opt_def)
    trainer = pl.Trainer(max_epochs=50, logger=logger)
    trainer.fit(model, data_train, data_val)



#%% 
##### FINAL MODEL #####
mc = ModelCheckpoint( every_n_epochs=1, save_top_k = -1)
logger = TensorBoardLogger('lightning_logs', name='final', )
model = DenseNN(2, 64, 0.3, 'LeakyReLU', 'Adam')
trainer = pl.Trainer(max_epochs=75, 
                     logger=logger, 
                     callbacks=[mc])
trainer.fit(model, data_train, data_val)


# %%
y_data = []
y_hat_data = []
for i, (x, y) in enumerate(data_test):
    preds = model.encoder(x.to('cuda'))
    y_hat_numpy = preds.detach().cpu().numpy().flatten()
    y_numpy = y.detach().cpu().numpy().flatten()

    y_data.extend(y_numpy)
    y_hat_data.extend(y_hat_numpy)

y_data = np.array(y_data)
y_hat_data = np.array(y_hat_data).round(0)
# %%
