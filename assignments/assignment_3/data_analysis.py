#%%
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

#%% 
rootdir = '/Users/konstantin/Documents/Projects/McGill/McGill-MGSC-673-AI/assignments/assignment_3/hyperpar_tune_2023-06-11_01-51-02'
res_t = pd.DataFrame()
count = 0 
for subdir, dirs, files in os.walk(rootdir):
    if 'progress.csv' in files:
        res = pd.read_csv(os.path.join(subdir, 'progress.csv'))
        res['iter'] = count
        res['vals'] = subdir.split('/')[-1]

        res_t = pd.concat([res_t, res])
        count += 1

# %% Fig 1
fig1dt = res_t.query('training_iteration == 300')[['Overall_val_','iter']]
fig1dt = fig1dt.sort_values('Overall_val_').iloc[:3,:].iter.to_list()
fig1dt = res_t.query(f'iter in {fig1dt}').sort_values('training_iteration')

fig1d = fig1dt.query('iter == 28')
fig1 = go.Figure(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_val_,
                           line={'color':'rgb(27,158,119)', 
                                 'width':1.5},
                           name=f'VALIDATION: Best'))
fig1.add_trace(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_train_,
                           line={'color':'rgb(27,158,119)',
                                 'width':1.5,
                                 'dash': 'dot'},
                           name=f'TRAIN: Best'))

fig1d = fig1dt.query('iter == 51')
fig1.add_trace(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_val_,
                           line={'color':'rgb(217,95,2)', 
                                 'width':0.5,},
                           name=f'VALIDATION: 2nd'))
fig1.add_trace(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_train_,
                           line={'color':'rgb(217,95,2)', 
                                 'width':0.5,
                                 'dash': 'dot'},
                           name=f'TRAIN: 2nd'))

fig1d = fig1dt.query('iter == 1')
fig1.add_trace(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_val_,
                           line={'color':'rgb(117,112,179)', 
                                 'width':0.5,
                                 'dash':'dash'},
                           name=f'VALIDATION: 3rd'))
fig1.add_trace(go.Scatter(x=fig1d.training_iteration, 
                           y=fig1d.Overall_train_,
                           line={'color':'rgb(117,112,179)', 
                                 'width':1,
                                 'dash': 'dot'},
                           name=f'TRAIN: 3rd'))
fig1.update_layout(template='none',
                  title={'text': "Overall Loss",},)
fig1.show(renderer='browser')


# %% Fig 2
fig2d = res_t.query(f'iter == 28').sort_values('training_iteration')

fig2 = go.Figure(go.Scatter(x=fig2d.training_iteration, 
                           y=fig2d.SalePrice_val_mse,
                           line={'color':'rgb(27,158,119)', 
                                 'width':1},
                           name=f'Sale Price MSE'))
fig2.add_trace(go.Scatter(x=fig2d.training_iteration, 
                           y=fig2d.YearRemod_val_mse,
                           line={'color':'rgb(217,95,2)', 
                                 'width':1,},
                           name=f'Year Remodelled MSE'))
fig2.add_trace(go.Scatter(x=fig2d.training_iteration, 
                           y=fig2d.YearBuilt_val_mse,
                           line={'color':'rgb(117,112,179)', 
                                 'width':1,},
                           name=f'Year Built MSE'))
fig2.add_trace(go.Scatter(x=fig2d.training_iteration, 
                           y=fig2d.BldgType_val_ce,
                           line={'color':'rgb(231,41,138)', 
                                 'width':1,},
                           name=f'Building Type CE'))
fig2.add_trace(go.Scatter(x=fig2d.training_iteration, 
                           y=fig2d.HouseStyle_val_ce,
                           line={'color':'rgb(102,166,30)', 
                                 'width':1,},
                           name=f'House Style CE'))


fig2.update_layout(template='none',title={'text': "Individual Loss",},)
fig2.show(renderer='browser')

# %%
