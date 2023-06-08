#%%
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl

import sklearn.preprocessing
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

##### DATASET DEFINITION #####
class PytorchData(Dataset):
    """Customer Churn Dataset."""

    def __init__(self, data_file, transform=None):
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
                    'GarageCond', 'PavedDrive',  
                    'SaleType',  'SaleCondition', 'Utilities',
                    
                    'MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd']
        self.emb_cols = ['MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd']

        self.remove_cols = ['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence']
        self.targ_cols = ['HouseStyle', 'BldgType', 'YearBuilt', 'YearRemodAdd', 'SalePrice']
        self.targ_cat = ['HouseStyle', 'BldgType']
        self.targ_num = ['YearBuilt', 'YearRemodAdd', 'SalePrice']


    def transform_data(self):
        # Input Data
        num_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='median')),
                             ('transform', sklearn.preprocessing.StandardScaler()),])
        cat_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='constant', 
                                                                     fill_value='NA')),
                             ('transform', sklearn.preprocessing.OneHotEncoder(drop='first')),])
        self.tr_pipe = ColumnTransformer([("numerical", num_pipe, self.num_cols),
                                          ("categorical", cat_pipe, self.cat_cols), ])
        self.tr_pipe.fit(self.df)
        df_x = self.tr_pipe.transform(self.df).toarray()
        df_x = pd.DataFrame(df_x, columns=self.tr_pipe.get_feature_names_out())
        self.df_x = df_x

        # Output
        self.targ_cat_pipe = Pipeline([('transform', sklearn.preprocessing.OneHotEncoder(drop='first'))])
        self.targ_num_pipe = Pipeline([('scale', sklearn.preprocessing.StandardScaler())])
        self.targ_cat_pipe.fit(self.df[self.targ_cat])
        self.targ_num_pipe.fit(self.df[self.targ_num])
        self.targ_pipe = ColumnTransformer([('target_cat', self.targ_cat_pipe, self.targ_cat),
                                            ('target_num', self.targ_num_pipe, self.targ_num)])
        df_y = self.targ_pipe.fit_transform(self.df).toarray()
        df_y =  pd.DataFrame(df_y, columns=self.targ_pipe.get_feature_names_out())
        self.df_y = df_y

    def transform_back_targ(self, y):
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
        y_vals = torch.tensor(self.df_y.iloc[idx, ], dtype=torch.float32)

        return (x_vals, y_vals)

##### MODEL DEFINITION #####
class DenseNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        shared = torch.nn.Sequential()
        regr = torch.nn.Sequential()
        clasf = torch.nn.Sequential()
        # self.optimizer_type = 
    
        # Shared Layer
        shared.append(torch.nn.Linear(248, 192))
        shared.append(torch.nn.Dropout(0.7))
        shared.append(torch.nn.ReLU())
        
        shared.append(torch.nn.Linear(192, 192))
        shared.append(torch.nn.Dropout(0.7))
        shared.append(torch.nn.ReLU())

        # Hidden Layers
        # for i in range(hd-1):  
        #     encoder.append(torch.nn.Linear(ls, ls))
        #     encoder.append(torch.nn.Dropout(dr))
        #     if af == 'ReLU': encoder.append(torch.nn.ReLU())
        #     elif af =='LeakyReLU': encoder.append(torch.nn.LeakyReLU())
        #     elif af =='ELU': encoder.append(torch.nn.ELU())
        #     elif af =='Tanh': encoder.append(torch.nn.Tanh())

        # Regression Output Layer
        regr.append(torch.nn.Linear(192, 3))
        
        # Classification Output Layer
        clasf.append(torch.nn.Linear(192, 11))
        clasf.append(torch.nn.Sigmoid())
        # shared.append(torch.nn.Dropout(dr))

        self.shared = shared
        self.regr = regr
        self.clasf = clasf

    def forward(self, x):
        shared_res = self.shared(x)
        regr_res = self.regr(shared_res)
        clasf_res = self.clasf(shared_res)
        final_res = torch.cat([clasf_res, regr_res], 1)
        return final_res

    def configure_optimizers(self):
        # if self.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # elif self.optimizer_type == 'SGD':
        #     optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        # elif self.optimizer_type == 'LBFGS':
        #     optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # Get Predictions
        x, y = train_batch
        res = self.forward(x)

        # Get Sale Price Prediction
        sale_price = y[:,-1]
        sale_price_pred = res[:,-1]
        sale_price_loss = torch.nn.functional.mse_loss(sale_price_pred, sale_price)
        self.log('regr_train_loss', sale_price_loss, prog_bar=True)

        # sale_price_unscaled = data.transform_back_targ(y.cpu().detach().numpy())[:,-1].astype('float32')
        # sale_price_pred_unscaled = data.transform_back_targ(res.cpu().detach().numpy())[:,-1].astype('float32')
        # sale_price_loss_scaled = torch.nn.functional.mse_loss(torch.tensor(sale_price_pred_unscaled), 
        #                                                       torch.tensor(sale_price_unscaled)) ** (1/2)
        # self.log('regr_uscaled_train_loss', sale_price_loss_scaled)

        # Get House Category Prediction
        return sale_price_loss

    def validation_step(self, valid_batch, batch_idx):
        # Get Predictions
        x, y = valid_batch
        res = self.forward(x)

        # Get Sale Price Prediction
        sale_price = y[:,-1]
        sale_price_pred = res[:,-1]
        sale_price_loss = torch.nn.functional.mse_loss(sale_price_pred, sale_price)
        self.log('regr_valid_loss', sale_price_loss, prog_bar=True)


        # sale_price_unscaled = data.transform_back_targ(y.cpu().detach().numpy())[:,-1].astype('float32')
        # sale_price_pred_unscaled = data.transform_back_targ(res.cpu().detach().numpy())[:,-1].astype('float32')
        # sale_price_loss_scaled = torch.nn.functional.mse_loss(torch.tensor(sale_price_pred_unscaled), 
        #                                                       torch.tensor(sale_price_unscaled)) ** (1/2)
        # self.log('regr_unscaled_valid_loss', sale_price_loss_scaled)

        # Get House Category Prediction
        return sale_price_loss



##### DATA PREPARATION #####
data = PytorchData('data/train.csv')
data_train, data_val = torch.utils.data.random_split(data, [1000, 460])
data_train = DataLoader(data_train, batch_size=200)
data_val = DataLoader(data_val, batch_size=200)

##### BASELINE MODEL #####
model = DenseNN()
trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model, data_train, data_val)
# %%
