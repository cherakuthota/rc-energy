# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import streamlit as st
from IPython import get_ipython

# %%
# Importing required libraries
import numpy as np
import pandas as pd
#pd.set_option('display.float_format', lambda x: '%.4f' % x)

import datetime as dt
import time

import math

#from scipy import stats

#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.stattools import pacf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from keras_tuner import RandomSearch
#from keras_tuner.engine.hyperparameters import HyperParameters

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

#sns.set_context("paper", font_scale=1.3)
#sns.set_style('white')
#
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff



import warnings
warnings.filterwarnings('ignore')
#!pip install keras-tuner -q
#!pip install keras-tuner --upgrade
# defining the user defined colors
clrs = ['#FAB800','#93BE3E','#35B179','#009A9A','#4E7E91']

st.title("Day ahead of You!")
st.header("wind energy")

# %% [markdown]
# # Loading data
# %% [markdown]
# display(dbutils.fs.ls("FileStore/tables"))
# %% [markdown]
# file_location = "/FileStore/tables/Actual_EQ.csv"
# file_type = "csv"
# %% [markdown]
# df = spark.read.csv(file_location,header = True, inferSchema=True)
# df = df.toPandas()

# %%



# %%
df = pd.read_csv('../data/Actual_EQ.csv')


# %%
# check the size of dataset
print(f'Dataset has {df.shape[0]} rows and {df.shape[1]} columns')
print(f"columns in raw dataset: ", list(df.columns))


# %%
cols= ['ds','act_price','pre_price']
df.columns = cols
# ds data type is converting to datetime type
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
# ds column is set as index
df.reset_index(drop=True,inplace=True)
#df.head(2)


# %%
print('The time series starts from : ', df.ds.min())
print('The time series ends on :     ', df.ds.max())
print(f'intervals of the dataset :   ', df.ds[1] - df.ds[0])


# %%
df=df.loc[:,['ds','pre_price','act_price']]
df.sort_values('ds', inplace=True, ascending=True)
df = df.reset_index(drop=True)
#df.head(2)


# %%
df = df.query("ds>='2021-01-01 00:00:00'")
df.reset_index(drop=True,inplace=True)
#df.head(2)

# %% [markdown]
# # data split
# %% [markdown]
# ## train and test data split

# %%
#Create training and test dataset. Training dataset is 80% of the total data and the remaining 20% will be predicted"""
lookBack = 168   # 1 week = 7*24 = 168hours considered to predict
n_ahead = 1      # forcast price at 1 hours ahead
train_size = int(len(df) * 0.80) #80% of data used for training
test_size = len(df) - train_size
df_train, df_test = df[0:train_size], df[train_size-lookBack-1:]
print(df_train.shape, df_test.shape, df_test.shape)
print('The train data starts from :     ', df_train.ds.min())
print('The train data ends on :         ', df_train.ds.max())
print(f'intervals of the train dataset :', df_train.ds[1] - df_train.ds[0])
print('The test data starts from :      ', df_test.ds.min())
print('The test data ends on :          ', df_test.ds.max())
print(f'intervals of the test dataset : ', df_test.ds[train_size-lookBack+1+1] - df_test.ds[train_size-lookBack+1])

# %% [markdown]
# ## splitting the dates from detasets

# %%
train_dates = pd.to_datetime(df_train['ds'])
test_dates = pd.to_datetime(df_test['ds'])


# %%
df_train = df_train.drop(['ds','pre_price'],axis = 1)
df_test = df_test.drop(['ds','pre_price'],axis = 1)


# %%
train_data = df_train.reset_index()['act_price']
test_data = df_test.reset_index()['act_price']
print('train_data shape:', train_data.shape)
print('test_data shape:', test_data.shape)

# normalization of data
scaler=MinMaxScaler(feature_range=(-1,1))
train_data=scaler.fit_transform(np.array(train_data).reshape(-1,1))
test_data=scaler.transform(np.array(test_data).reshape(-1,1))
#
# features and target data set split
# convert an array of values into a dataset matrix
def create_datasets(dataset,target_index,lookback = 1,n_ahead = 0):
    X, y = [], []
    for i in range(lookback, dataset.shape[0]- n_ahead+1):
        X.append(dataset[i-lookback:i,0:dataset.shape[1]])
        y.append(dataset[i+n_ahead-1:i+n_ahead,target_index])
    return np.array(X), np.array(y)
#
# Splitting the dataset into features and targets
X_train, y_train= create_datasets(train_data, target_index = 0,lookback = lookBack,n_ahead = n_ahead)
X_test, y_test = create_datasets(test_data, target_index = 0,lookback = lookBack,n_ahead = n_ahead)
# print(X_train.shape), print(y_train.shape)

# # building LSTM regression model
# create model framework and fit the LSTM network
#model=Sequential()
#model.add(LSTM(100, activation ='relu', input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences = True))
#model.add(Dropout(0.2))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(50,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(50))
#model.add(Dropout(0.2))
#model.add(Dense(1))

#
#"""Compile and fit the model"""
#model.compile(loss='mean_squared_error',optimizer='adam')
#stop_early = EarlyStopping(monitor='val_loss', patience=3)
#model.summary()
#history = model.fit(X_train,y_train,validation_split=0.2,epochs=50,batch_size=24,callbacks=[stop_early],verbose=1,shuffle=False)

# %%
# Saving model to disk
import pickle
#pickle.dump(model, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# %%
### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

# %%
##Transform back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)


# %%
st.title("Model Evaluation")
st.markdown("##")
train_rsme  = np.sqrt(mean_squared_error(y_train, train_predict)).round(2)
test_rsme   = np.sqrt(mean_squared_error(y_test, test_predict)).round(2)

train_mse   = mean_squared_error(y_train, train_predict).round(2)
test_mse    = mean_squared_error(y_test, test_predict).round(2)

## TOP KPI's
left_column, mid1_column, mid2_column,right_column = st.columns(4)
with left_column:
    st.subheader("Train model RSME:")
    st.subheader(f"{train_rsme:,}")
with mid1_column:
    st.subheader("Train model MSE:")
    st.subheader(f"{train_mse:,}")
with mid2_column:
    st.subheader("Test model RSME:")
    st.subheader(f"{test_rsme:,}")
with right_column:
    st.subheader("Test model MSE:")
    st.subheader(f"{test_mse:,}")

# %%
print(test_data.shape)
x_input=test_data[len(test_data)-lookBack:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# %%
from numpy import array

lst_output=[]
n_steps=lookBack
i=0
while(i<=24):
    
    if(len(temp_input)>lookBack):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print(f"{i} hour input {x_input}")#.format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print(f"{i} hour output {yhat}")#.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
lst_output = scaler.inverse_transform(lst_output)

# %%
future_time = pd.date_range(list(test_dates)[-1],periods = 24,freq='1H')#.tolist()
future_time = pd.DataFrame(future_time)[:]

# %%
df_forecast = pd.merge(future_time,pd.DataFrame(lst_output[0:-1]),left_index = True,right_index = True,how='left')
df_forecast.columns = ['ds','lstm_pred']
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], format='%Y-%m-%d %H:%M:%S')


# %%
df_forecast.set_index('ds').to_csv('forcasted_prices.csv')

st.markdown("""---""")
st.title("Forecasted Prices")
st.write(df_forecast)
# %%
df_train_predicted = pd.DataFrame({'ds':train_dates[lookBack:].values,'lstm_pred':train_predict[:,0]})
df_test_predicted =pd.DataFrame({'ds':test_dates[lookBack:].values,'lstm_pred':test_predict[:,0]})

# %%
df_test_predicted['y_test'] = y_test[:,0]
df_test_predicted['residual']  = df_test_predicted['y_test'] - df_test_predicted['lstm_pred'] 


# %%
df_train_predicted['y_train'] = y_train[:,0]
df_train_predicted['residual']  = df_train_predicted['y_train'] - df_train_predicted['lstm_pred'] 

st.markdown("""---""")
st.title("Forecasted Prices")
# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df["act_price"],
              name="actual price", line=dict(color=clrs[0], width=4)))
fig.add_trace(go.Bar(x=df_train_predicted['ds'][:-1], y=df_train_predicted['lstm_pred'][:-1], name = 'train predicted price', marker_color=clrs[2]))
fig.add_trace(go.Bar(x=df_test_predicted['ds'][:-1], y=df_test_predicted['lstm_pred'][:-1], name = 'test predicted price', marker_color=clrs[4]))
fig.add_trace(go.Bar(x=df_forecast['ds'], y=df_forecast['lstm_pred'], name = 'forecasted price', marker_color=clrs[3]))


fig.update_layout(
    title='actual and predicted prices using LSTM',
    xaxis_nticks=25,
    yaxis_nticks=10,
    xaxis_title=" date time",
    yaxis_title=" Price in €/mwh",
    autosize=False,
    width=1000,
    height=600,
    legend_title="prices",
    font=dict(size=10),
    legend=dict(yanchor="top", y=0.3, xanchor="left", x=0.01)
)
fig.update_xaxes(
    tickangle=-45,
    title_text="Date time",
    title_font={"size": 12},
    title_standoff=25,
    tickformat='%Y-%m-%d<br>%H:%M')

fig.update_yaxes(
    title_text="price €/mwh",
    title_standoff=25)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=8, label="8H", step="hour", stepmode="backward"),
            dict(count=1, label="1D", step="day", stepmode="backward"),
            dict(count=2, label="2D", step="day", stepmode="backward"),
            dict(count=1, label="1w", step="month", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=2, label="2m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ))

st.plotly_chart(fig)
st.markdown("""---""")

# %% [markdown]
st.title('Residuals')

# %%
fig = go.Figure()
fig.add_trace(go.Bar(x=df_train_predicted['ds'], y=df_train_predicted["residual"], name = 'lstm model residuals', marker_color=clrs[2]))
fig.add_trace(go.Bar(x=df_test_predicted['ds'], y=df_test_predicted["residual"], name = 'lstm model residuals', marker_color=clrs[4]))
fig.add_trace(go.Bar(x=df['ds'], y=df["act_price"]-df["pre_price"], name = 'original model residuals', marker_color=clrs[0]))
fig.update_layout(
    title='actual and predicted residuals',
    xaxis_nticks=25,
    yaxis_nticks=10,
    xaxis_title=" date time",
    yaxis_title=" residual in price €/mwh",
    autosize=True,
    width=1000,
    height=600,
    legend_title="price residual",
    font=dict(size=10),
    legend=dict(yanchor="top", y=1.0, xanchor="right", x=0.98),
    yaxis_range = [-5,30]
)
fig.update_xaxes(
    tickangle=-45,
    title_text="Date time",
    title_font={"size": 16},
    title_standoff=25,
    tickformat='%Y-%m-%d<br>%H:%M')

fig.update_yaxes(
    title_text="price €/mwh",
    title_standoff=10
    )

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=8, label="8H", step="hour", stepmode="backward"),
            dict(count=2, label="2D", step="day", stepmode="backward"),
            dict(count=1, label="1w", step="month", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    ))

st.plotly_chart(fig)
st.markdown("""---""")

# %%
fig = px.scatter(x= df.act_price, y = df.pre_price,labels={'x':'Actual price','y':'Predicted price'},opacity=0.65,
                 trendline='ols', trendline_color_override=clrs[0],
                 title='Previous predictions')
fig.update_layout(
    title='Existing Model Residuals ',
    xaxis_nticks=25,
    yaxis_nticks=10,
    xaxis_title=" date time",
    yaxis_title=" Price in €/mwh",
    autosize=False,
    width=1000,
    height=600,
    legend_title="prices",
    font=dict(size=10),
    legend=dict(yanchor="top", y=0.4, xanchor="left", x=0.01))
st.plotly_chart(fig)
st.markdown("""---""")

# %%
fig = px.scatter(x= df_train_predicted.y_train, y = df_train_predicted.lstm_pred,labels={'x':'Actual price','y':'Predicted price'},opacity=0.65,
                 trendline='ols', trendline_color_override=clrs[0],
                 title='Trained Model Residuals'
)
fig.update_layout(width=1000,height=600,)
st.plotly_chart(fig)
st.markdown("""---""")

# %%
fig = px.scatter(x= df_test_predicted.y_test, y = df_test_predicted.lstm_pred,labels={'x':'Actual price','y':'Predicted price'},opacity=0.65,
                 trendline='ols', trendline_color_override=clrs[0],
                 title='Test Model Residuals'
                 )
fig.update_layout(width=1000,height=600,)
st.plotly_chart(fig)
st.markdown("""---""")
#
## %% [markdown]
#def oppertunity_analyses(dataset, dataset_name,col_y, col_yhat):
#    df = dataset.copy()
#    df['error'] = df[col_y]-df[col_yhat]
#    total = int(df[col_y].sum().round(2))
#    opportunity = int(abs(df.query("error < 0")[col_yhat].sum()))
#    opportunity = round(opportunity,2)
#    overbidding = abs(df.query("error >= 0")[col_y].sum()).round(2)
#    income = (total - opportunity - overbidding).round(2)
#    mse = mean_squared_error(df[col_y],df[col_yhat]).round(2)
#    rsme = abs(mse**0.5).round(2)
#
#    df_temp = pd.DataFrame(
#        [[total,opportunity,overbidding,income,mse,rsme]],columns = ['total','opportunity','overbidding','income','mse','rsme'],index=[dataset_name])
#    #print (df_temp)
#    return df_temp#[total,opportunity,overbidding,income,mse,rsme]
#
#
## %%
#df_final_evaluation = pd.DataFrame(columns = ['total','opportunity','overbidding','income','mse','rsme'])
#
#
## %%
#df_final_evaluation = df_final_evaluation.append(oppertunity_analyses(df,'Base Model','act_price','pre_price'))
#df_final_evaluation = df_final_evaluation.append(oppertunity_analyses(df_train_predicted,'Trained Model','y_train','lstm_pred'))
#df_final_evaluation = df_final_evaluation.append(oppertunity_analyses(df_test_predicted,'Test Model','y_test','lstm_pred'))
#
#
## %%
#df_final_evaluation= df_final_evaluation.reset_index()
#
#
## %%
#cols = df_final_evaluation.columns[1:]
#
## %%
#df2 = df_final_evaluation.copy()
#
## %%
#for col in list(cols):
#    print(col)
#    #df2[col]
#    df2[col] = df2[col]/df_final_evaluation['total']*1000000
#df2
#
#
## %%
#x = list(df2["index"].values)
#fig = go.Figure(go.Bar(x=x, y=df2['opportunity'], name='Opportunity',marker_color=clrs[4]))
#fig.add_trace(go.Bar(x=x, y=df2['overbidding'], name='Overbiddin',marker_color=clrs[0]))
#fig.add_trace(go.Bar(x=x, y=-df2['income'], name='Income',marker_color=clrs[3]))
#fig.update_layout(barmode='stack', width = 1000, height=600)
#st.plotly_chart(fig)
#st.markdown("""---""")
#st.title("THANK YOU")
#
#
## %%
##df2.plot()