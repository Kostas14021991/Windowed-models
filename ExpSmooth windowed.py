# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:52:52 2023

@author: kosta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sktime.forecasting.model_selection import temporal_train_test_split
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error

path = "C:\\Users\kosta\\Documents\\PhD related\\Python\\Feature forecasting\\All metals.xlsx"

df = pd.read_excel(path,parse_dates=['DATE'],index_col='DATE')
df.head()

# Column 8 gives good results


df = df.iloc[:,0].to_frame()
df.head()

     

def windower(series,window):
    
        df1 = pd.DataFrame(index=df.index[window:],columns=[str(i) for i in range(1,window+1)])
        df1['target'] = np.nan
 
        for i in range(len(df)-window):
     
            df1.iloc[i,:window] = df.iloc[i:window+i].values.flatten()
            df1['target'].iloc[i] = df.iloc[i+window]
        
        return df1

windowed = windower(df,15)


def exp_forecast_preds(series,step=1):
    #fh = np.arange(1,2)
    model = ExponentialSmoothing(series).fit()
    pred_one_step = model.forecast(step)
    alpha = model.params['smoothing_level']
    return pred_one_step.values, alpha

df


exp_forecast_preds(df)

l = []
k = []
naive = []
alphas = []
window = 5
for i in range(len(df)-window):
    
    f,a = exp_forecast_preds(df.iloc[i:window+i])
    alphas.append(a)
    r = df.iloc[i+window]
    naive.append(df.iloc[i+window-1])
    l.append(f)
    k.append(r)

plt.plot(l[:100],label='forecast',marker='o',markersize=3)
plt.plot(k[:100],label='real',marker='o',markersize=3)
plt.plot(naive[:100],label='naive',marker='o',markersize=3)
plt.title(f'RMSE Model {df.columns[0]} = {np.sqrt(mean_squared_error(k,l))}, Naive = {np.sqrt(mean_squared_error(k,naive))}')
plt.legend()
plt.show()
print()

plt.plot(alphas)



error = []
errornaive = []
#window = 7
for window in range(3,50):
    l = []
    k = []
    naive = []
    for i in range(len(df)-window):
    
        f,_ = exp_forecast_preds(df.iloc[i:window+i])
        r = df.iloc[i+window]
        l.append(f)
        k.append(r.values)
        naive.append(df.iloc[i+window-1].values)
    error.append(mean_squared_error(k,l))
    errornaive.append(mean_squared_error(k,naive))
        
plt.plot(range(3,50),np.sqrt(error),label='theta')
plt.plot(range(3,50),np.sqrt(errornaive),label='naive')
plt.plot(range(3,50),np.sqrt((np.array(errornaive) + np.array(error))/2),label='combined')
plt.legend()
plt.show()
















