#%%
import pandas as pd
import plotly.express as px

import sklearn.preprocessing
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

import os

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


model = DenseNN.load_from_checkpoint(os.path.join('lightning_logs','final', 'version_0', 'checkpoints', 'test.ckpt'))



#%% 
# LAYER SIZE
option = ['ls_32', 'ls_64', 'ls_128', 'ls_256', 'ls_512']
res = pd.DataFrame()
for opt_n in option:
    opt = opt_n
    if opt_n == 'ls_256':
        opt = 'baseline'

    train = pd.read_csv(f"results/run-{opt}_version_0-tag-train_loss.csv")
    validation = pd.read_csv(f"results/run-{opt}_version_0-tag-val_loss.csv")
    
    train['Param'] = opt_n
    train['Type'] = 'Train Data'
    validation['Param'] = opt_n
    validation['Type'] = 'Test Data'
    res = pd.concat([res, validation])

color_scheme = ['rgba(27,158,119,0.3)', 'rgba(217,95,2)',
                'rgba(117,112,179, 0.3)', 'rgba(231,41,138,0.3)',
                'rgba(102,166,30,0.3)', 'rgba(230,171,2)',
                'rgba(166,118,29)', 'rgba(102,102,102)']
fig = px.line(res, x='Step', y='Value', color='Param', line_dash='Type', 
              template='none',
              color_discrete_sequence=color_scheme,
              title='Hidden Layer Size')
fig.show(renderer='browser')
res.sort_values('Value')






#%%
# NUMBER OF HIDDEN LAYERS
option = ['hl_1', 'hl_2', 'hl_3']
res = pd.DataFrame()
for opt_n in option:
    opt = opt_n
    if opt_n == 'hl_1':
        opt = 'baseline'

    train = pd.read_csv(f"results/run-{opt}_version_0-tag-train_loss.csv")
    validation = pd.read_csv(f"results/run-{opt}_version_0-tag-val_loss.csv")
    
    train['Param'] = opt_n
    train['Type'] = 'Train Data'
    validation['Param'] = opt_n
    validation['Type'] = 'Test Data'
    res = pd.concat([res, validation])

color_scheme = ['rgba(27,158,119,0.3)', 'rgba(217,95,2)',
                'rgba(117,112,179, 0.3)', 'rgba(231,41,138,0.3)',
                'rgba(102,166,30,0.3)', 'rgba(230,171,2)',
                'rgba(166,118,29)', 'rgba(102,102,102)']
fig = px.line(res, x='Step', y='Value', color='Param', line_dash='Type', 
              template='none',
              color_discrete_sequence=color_scheme,
              title='Number of Hidden Layers')
fig.show(renderer='browser')
res.sort_values('Value')






#%%
# OPTIMIZER
option = ['opt_SGD', 'opt_Adam', 'opt_LBFGS']
res = pd.DataFrame()
for opt_n in option:
    opt = opt_n
    if opt_n == 'opt_Adam':
        opt = 'baseline'

    train = pd.read_csv(f"results/run-{opt}_version_0-tag-train_loss.csv")
    validation = pd.read_csv(f"results/run-{opt}_version_0-tag-val_loss.csv")
    
    train['Param'] = opt_n
    train['Type'] = 'Train Data'
    validation['Param'] = opt_n
    validation['Type'] = 'Test Data'
    res = pd.concat([res, validation])

color_scheme = ['rgba(27,158,119,0.3)', 'rgba(217,95,2)',
                'rgba(117,112,179, 0.3)', 'rgba(231,41,138,0.3)',
                'rgba(102,166,30,0.3)', 'rgba(230,171,2)',
                'rgba(166,118,29)', 'rgba(102,102,102)']
fig = px.line(res, x='Step', y='Value', color='Param', line_dash='Type', 
              template='none',
              color_discrete_sequence=color_scheme,
              title='Optimizers')
fig.show(renderer='browser')
res.sort_values('Value')






#%%
# DROPOUT RATE
option = ['dr_0.0', 'dr_0.1', 'dr_0.2', 'dr_0.25', 'dr_0.3', 'dr_0.4', 'dr_0.5', 'dr_0.6']
res = pd.DataFrame()
for opt_n in option:
    opt = opt_n
    if opt_n == 'dr_0.25':
        opt = 'baseline'

    train = pd.read_csv(f"results/run-{opt}_version_0-tag-train_loss.csv")
    validation = pd.read_csv(f"results/run-{opt}_version_0-tag-val_loss.csv")
    
    train['Param'] = opt_n
    train['Type'] = 'Train Data'
    validation['Param'] = opt_n
    validation['Type'] = 'Test Data'
    res = pd.concat([res, validation])

color_scheme = ['rgba(27,158,119,0.3)', 'rgba(217,95,2,0.3)',
                'rgba(117,112,179, 0.3)', 'rgba(231,41,138,0.3)',
                'rgba(102,166,30)', 'rgba(230,171,2,0.3)',
                'rgba(166,118,29,0.3)', 'rgba(102,102,102,0.3)',]
fig = px.line(res, x='Step', y='Value', color='Param', line_dash='Type', 
              template='none',
              color_discrete_sequence=color_scheme,
              title='Dropout Rate')
fig.show(renderer='browser')
res.sort_values('Value')






#%%
# ACTIVATION FUNCTIONS
option = ['act_ELU', 'act_ReLU', 'act_LeakyReLU', 'act_Tanh']
res = pd.DataFrame()
for opt_n in option:
    opt = opt_n
    if opt_n == 'act_ReLU':
        opt = 'baseline'

    train = pd.read_csv(f"results/run-{opt}_version_0-tag-train_loss.csv")
    validation = pd.read_csv(f"results/run-{opt}_version_0-tag-val_loss.csv")
    
    train['Param'] = opt_n
    train['Type'] = 'Train Data'
    validation['Param'] = opt_n
    validation['Type'] = 'Test Data'
    res = pd.concat([res, validation])

color_scheme = ['rgba(27,158,119,0.3)', 'rgba(217,95,2,0.3)',
                'rgba(117,112,179)', 'rgba(231,41,138,0.3)',
                'rgba(102,166,30)', 'rgba(230,171,2,0.3)',
                'rgba(166,118,29,0.3)', 'rgba(102,102,102,0.3)',]
fig = px.line(res, x='Step', y='Value', color='Param', line_dash='Type', 
              template='none',
              color_discrete_sequence=color_scheme,
              title='Activation Functions')
fig.show(renderer='browser')
res.sort_values('Value')
# %%
