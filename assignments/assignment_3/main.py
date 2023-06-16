#%%
import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune, air
from ray.tune.search.optuna import OptunaSearch

import sklearn.preprocessing
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

##### DATASET DEFINITION #####
class PytorchData(pl.LightningDataModule):
    """Customer Churn Dataset."""

    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.__get_col_types__()
        self.transform_data()
    
    def __get_col_types__(self):
        """Identifies Numerical, Categorical, and Target Columns"""
        self.num_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                    'MasVnrArea', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                    'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                    'MiscVal', 'MoSold', 'YrSold']
        self.cat_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 
                    'Utilities', 'LotConfig', 'LandSlope', 'Condition1',
                    'Condition2', 'RoofStyle',
                    'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond',
                    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                    'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                    'GarageCond', 'PavedDrive', 'Fence', 'Alley',
                    'SaleType',  'SaleCondition', 'Utilities', 'MiscFeature',
                    'Neighborhood', 'Exterior1st', 'Exterior2nd']
        self.remove_cols = ['Id', 'MSSubClass', 'PoolQC']
        
        self.targ_cols = ['HouseStyle', 'BldgType', 'YearBuilt', 'YearRemodAdd', 'SalePrice']
        self.targ_cat = ['HouseStyle', 'BldgType']
        self.targ_num = ['YearBuilt', 'YearRemodAdd', 'SalePrice']



    def transform_data(self):
        # Input Data
        num_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='median')),
                             ('transform', sklearn.preprocessing.StandardScaler()),])
        cat_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='constant', fill_value='NA')),
                             ('transform', sklearn.preprocessing.OneHotEncoder(drop='first')),]) 
        self.tr_pipe = ColumnTransformer([("numerical", num_pipe, self.num_cols), 
                                          ("categorical", cat_pipe, self.cat_cols), ])
        df_x = self.tr_pipe.fit_transform(self.df).toarray()
        df_x = pd.DataFrame(df_x, columns=self.tr_pipe.get_feature_names_out())
        self.df_x = df_x

        # Sale Prices
        self.spr_pipe = Pipeline([('scale', sklearn.preprocessing.StandardScaler())])
        df_spr = self.spr_pipe.fit_transform(self.df[['SalePrice']])
        self.df_spr = pd.DataFrame(df_spr, columns=self.spr_pipe.get_feature_names_out())

        # Year Remodelled
        self.yrm_pipe = Pipeline([('scale', sklearn.preprocessing.StandardScaler())])
        df_yrm = self.yrm_pipe.fit_transform(self.df[['YearRemodAdd']])
        self.df_yrm = pd.DataFrame(df_yrm, columns=self.yrm_pipe.get_feature_names_out())

        # Year Build
        self.ybl_pipe = Pipeline([('scale', sklearn.preprocessing.StandardScaler())])
        df_ybl = self.ybl_pipe.fit_transform(self.df[['YearBuilt']])
        self.df_ybl = pd.DataFrame(df_ybl, columns=self.ybl_pipe.get_feature_names_out())

        # Building Type
        self.btp_pipe = Pipeline([('scale', sklearn.preprocessing.OneHotEncoder(drop=None, sparse_output=False))])
        df_btp = self.btp_pipe.fit_transform(self.df[['BldgType']])
        self.df_btp = pd.DataFrame(df_btp, columns=self.btp_pipe.get_feature_names_out())

        # House Style
        self.hst_pipe = Pipeline([('scale', sklearn.preprocessing.OneHotEncoder(drop=None, sparse_output=False))])
        df_hst = self.hst_pipe.fit_transform(self.df[['HouseStyle']])
        self.df_hst = pd.DataFrame(df_hst, columns=self.hst_pipe.get_feature_names_out())

    def transform_back_targ(self, y):
        y = y.cpu().detach().numpy()
        cat_res = self.targ_cat_pipe.inverse_transform(y[:,:-3])
        num_res = self.targ_num_pipe.inverse_transform(y[:,-3:])
        targ = np.concatenate([cat_res, num_res], axis=1)
        return(targ)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_vals = torch.tensor(self.df_x.iloc[idx, ], dtype=torch.float32)
        spr_vals = torch.tensor(self.df_spr.iloc[idx, ], dtype=torch.float32)
        yrm_vals = torch.tensor(self.df_yrm.iloc[idx, ], dtype=torch.float32)
        ybl_vals = torch.tensor(self.df_ybl.iloc[idx, ], dtype=torch.float32)
        btp_vals = torch.tensor(self.df_btp.iloc[idx, ], dtype=torch.float32)
        hst_vals = torch.tensor(self.df_hst.iloc[idx, ], dtype=torch.float32)

        return (x_vals, spr_vals, yrm_vals, ybl_vals, btp_vals, hst_vals)

##### MODEL DEFINITION #####
class DenseNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Config
        self.hl = config['hl']
        self.ls = config['hls']
        self.do = config['dr']
        self.on = config['opt']
        self.lr = config['lr']
        self.act = config['act']

        shared = torch.nn.Sequential()
        spr = torch.nn.Sequential()
        yrm = torch.nn.Sequential()
        ybl = torch.nn.Sequential()
        btp = torch.nn.Sequential()
        hst = torch.nn.Sequential()
 
        # Input Layer
        shared.append(torch.nn.Linear(244, self.ls))
        shared.append(torch.nn.Dropout(self.do))
        if self.act == 'ReLU': shared.append(torch.nn.ReLU())
        if self.act == 'LRelu': shared.append(torch.nn.LeakyReLU())
        if self.act == 'Tanh': shared.append(torch.nn.Tanh())

        # shared.append(torch.nn.ReLU())

        # Hidden Layers
        for hl in range(self.hl-1):
            shared.append(torch.nn.Linear(self.ls, self.ls))
            shared.append(torch.nn.Dropout(self.do))
            if self.act == 'ReLU': shared.append(torch.nn.ReLU())
            if self.act == 'LRelu': shared.append(torch.nn.LeakyReLU())
            if self.act == 'Tanh': shared.append(torch.nn.Tanh())

        # Multiple Output Layers
        spr.append(torch.nn.Linear(self.ls, 1))
        yrm.append(torch.nn.Linear(self.ls, 1))
        ybl.append(torch.nn.Linear(self.ls, 1))

        btp.append(torch.nn.Linear(self.ls, 5))
        btp.append(torch.nn.Softmax())
        hst.append(torch.nn.Linear(self.ls, 8))
        hst.append(torch.nn.Softmax())

        self.shared = shared
        self.spr = spr
        self.yrm = yrm
        self.ybl = ybl
        self.btp = btp
        self.hst = hst

    def forward(self, x):
        shared_res = self.shared(x)

        sale_price = self.spr(shared_res)
        year_remod = self.yrm(shared_res)
        year_built = self.ybl(shared_res)
        build_type = self.btp(shared_res)
        house_style = self.hst(shared_res)
        return sale_price, year_remod, year_built, build_type, house_style

    def configure_optimizers(self):
        if self.on == 'Adam': optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.on == 'SGD': optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.on == 'LBFGS': optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Get Values and Predictions
        x, sale_price, year_remod, year_built, build_type, house_style = train_batch
        sale_price_pred, year_remod_pred, year_built_pred, build_type_pred, house_style_pred = self.forward(x)

        # Get Loss
        sale_price_loss = torch.nn.functional.mse_loss(sale_price_pred, sale_price)
        year_remod_loss = torch.nn.functional.mse_loss(year_remod_pred, year_remod)
        year_built_loss = torch.nn.functional.mse_loss(year_built_pred, year_built)
        build_type_loss = torch.nn.functional.cross_entropy(build_type_pred, build_type)
        house_style_loss = torch.nn.functional.cross_entropy(house_style_pred, house_style)
        tot_loss = sale_price_loss + year_remod_loss + year_built_loss + build_type_loss + house_style_loss

        # Logging
        self.log('SalePrice_train_mse', sale_price_loss)
        self.log('YearRemod_train_mse', year_remod_loss)
        self.log('YearBuilt_train_mse', year_built_loss)
        self.log('BldgType_train_ce', build_type_loss)
        self.log('HouseStyle_train_ce', house_style_loss)
        self.log('Overall_train_', tot_loss)
        return tot_loss

    def validation_step(self, valid_batch, batch_idx):
        # Get Values and Predictions
        x, sale_price, year_remod, year_built, build_type, house_style = valid_batch
        sale_price_pred, year_remod_pred, year_built_pred, build_type_pred, house_style_pred = self.forward(x)

        # Get Loss
        sale_price_loss = torch.nn.functional.mse_loss(sale_price_pred, sale_price)
        year_remod_loss = torch.nn.functional.mse_loss(year_remod_pred, year_remod)
        year_built_loss = torch.nn.functional.mse_loss(year_built_pred, year_built)
        build_type_loss = torch.nn.functional.cross_entropy(build_type_pred, build_type)
        house_style_loss = torch.nn.functional.cross_entropy(house_style_pred, house_style)
        tot_loss = sale_price_loss + year_remod_loss + year_built_loss + build_type_loss + house_style_loss
        
        # Logging
        self.log('SalePrice_val_mse', sale_price_loss,)
        self.log('YearRemod_val_mse', year_remod_loss)
        self.log('YearBuilt_val_mse', year_built_loss)
        self.log('BldgType_val_ce', build_type_loss)
        self.log('HouseStyle_val_ce', house_style_loss)
        self.log('Overall_val_', tot_loss)
        return tot_loss

##### HYPERPARAMETER TUNING #####
def hyperpar_tune(config, data_train):
    model = DenseNN(config)
    data_train = DataLoader(data_train, batch_size=config['bs'])

    metrics = ['SalePrice_val_mse', 'YearRemod_val_mse', 
               'YearBuilt_val_mse', 'BldgType_val_ce', 
               'HouseStyle_val_ce', 'Overall_val_',
               'SalePrice_train_mse', 'YearRemod_train_mse', 
               'YearBuilt_train_mse', 'BldgType_train_ce', 
               'HouseStyle_train_ce', 'Overall_train_',]
    logger = pl.loggers.TensorBoardLogger(save_dir=ORIG_DIR, name='inner')
    trainer = pl.Trainer(max_epochs=300, 
                         enable_progress_bar=False,
                         logger=logger,
                         callbacks=[TuneReportCallback(metrics, on="validation_end")])
    trainer.fit(model, data_train, data_val)

#%%
ORIG_DIR = "/Users/konstantin/Documents/Projects/McGill/McGill-MGSC-673-AI/assignments/assignment_3"

##### DATA PREPARATION #####
data = PytorchData('data/train.csv')
data_train, data_val, data_test = torch.utils.data.random_split(data, [1000, 400, 60])
data_val = DataLoader(data_val, batch_size=400)
data_test = DataLoader(data_test, batch_size=60)

##### HYPERPARAMETERS #####
config = {"hl": tune.randint(1,4),
          "hls": tune.randint(30, 500),
          "dr": tune.loguniform(1e-4, 0.9),
          'opt': tune.choice(['Adam', 'SGD' ,'LBFGS']),
          "lr": tune.loguniform(1e-4, 1e-1),
          'act': tune.choice(['ReLU', 'LRelu' ,'Tanh']),
          "bs": tune.randint(10, 501),}
algo = OptunaSearch()

tuner = tune.Tuner(tune.with_parameters(hyperpar_tune, data_train = data_train,),
                   param_space = config,
                   run_config=air.RunConfig(local_dir=ORIG_DIR),
                   tune_config=tune.TuneConfig(mode="min", 
                                               metric='Overall_val_',
                                               num_samples=100,
                                               max_concurrent_trials=7,
                                               search_alg=algo))
tuner.fit()