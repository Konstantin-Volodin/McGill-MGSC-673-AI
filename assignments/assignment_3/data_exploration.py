#%%
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

import sklearn.preprocessing
import sklearn.impute
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %% Review Data
df = pd.read_csv('data/train.csv')
df.info()

#%%
num_cols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
            'MasVnrArea', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
            'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
            'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'MiscVal', 'MoSold', 'YrSold']

cat_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 
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
emb_cols = ['MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd']

remove_cols = ['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence']
targ_cols = ['SalePrice', 'HouseStyle', 'BldgType', 'YearBuilt', 'YearRemodAdd']


#%%
cat_cols=['continent','position','sub_position']

for i in cat_cols:  
    
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])

    # Define the number of categories and the embedding size
    num_categories = data[i].nunique()
    embedding_dim = data[i].nunique()//2

    # Create a tensor of categorical values
    categorical_column = torch.randint(low=0, high=num_categories, size=(len(data[i]),))

    # Create the embedding layer
    embedding_layer = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_dim)

    # Apply the embedding layer to the categorical column
    embedded_column = embedding_layer(categorical_column)

    embedded_df = pd.DataFrame(embedded_column.detach().numpy(), columns=[i+f'_embed_{j}' for j in range(embedding_dim)])

    # Replace the original categorical column with the embedded DataFrame
    data = pd.concat([data.drop(i, axis=1), embedded_df], axis=1)


#%%
##### DATASET DEFINITION #####
class CustomerChurn(Dataset):
    """Customer Churn Dataset."""

    def __init__(self, data_file, transform=None):
        self.df = pd.read_csv(data_file)
        self.__get_col_types__()
        # self.__transform_data__()

    
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
        self.targ_cols = ['SalePrice', 'HouseStyle', 'BldgType', 'YearBuilt', 'YearRemodAdd']


    def __transform_data__(self):
        num_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='median')),
                             ('transform', sklearn.preprocessing.StandardScaler()),])
        cat_pipe = Pipeline([('impute', sklearn.impute.SimpleImputer(strategy='constant', 
                                                                     fill_value='NA')),
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

# %%
temp = CustomerChurn('data/train.csv')
# %%
