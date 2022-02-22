import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pandas_datareader
import yfinance as yf
from plotly import graph_objs as go
import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


START = '2015-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock price prediction system by Kingard")
stocks = ('AAPL','TSLA','MSFT','AMZN','BTC-USD','ETH-USD','XRP-USD','BCH-USD')
selected_stock = st.selectbox("Select the stock/financial instrument of choice:",stocks)
#period = n_years * 365  # in days

# Test price choice concept
price_types = ('Open','High','Low','Close')
price_of_choice = st.selectbox("Select the price of choice(Close price is recommended):",price_types)

@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data['% Returns'] = data[price_of_choice].pct_change()  # we find the percentage change using the pct_change() method
    data['Log returns'] = np.log(1 + data['% Returns'])  # from the percentage returns we can easily compute log returns
    data.dropna(inplace=True)
    return data


data_load_status = st.text("loading data...")
data = load_data(selected_stock)
data_load_status = st.text("loading data...Done!")

st.subheader('Raw data')
st.write(data.tail())


# Here we plot the raw data to have a sense of what it looks like
def plot_data():
    layout = go.Layout(showlegend=True)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=data.index, y=data[price_of_choice], name="stock {} price".format(price_of_choice)))
    fig.add_trace(go.Scatter(x=data.index, y=data[price_of_choice].rolling(100).mean(), name="100-day moving average"))
    fig.add_trace(go.Scatter(x=data.index, y=data[price_of_choice].rolling(200).mean(), name="200-day moving average"))
    fig.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
    fig.update_layout(
        width=1000,
        height=650
    )
    st.plotly_chart(fig)


# data_plot_button_val = st.button('plot data', key=None, help=None, on_click=plot_data())
plot_data()


# Here we plot the log returns and see if it mean-reverts at 0
def plot_extracted_data():
    layout = go.Layout(showlegend=True)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=data.index, y=data['Log returns'], name="log returns"))
    fig.layout.update(title_text="Computed Log returns", xaxis_rangeslider_visible=True)
    fig.update_layout(
        width=900,
        height=650
    )
    st.plotly_chart(fig)

plot_extracted_data()


#x = data[['Close','Log returns']]
# price_types = ('Open','High','Low','Close')
# price_of_choice = st.selectbox("Select the stock of choice(Close price is recommended):",price_types)


x = data[[price_of_choice,'Log returns']]
scaler = MinMaxScaler(feature_range=(0,1)).fit(x)
x_scaled = scaler.transform(x)
y = [x[0] for x in x_scaled]
split_point = int(len(x_scaled)*0.8)
X_train = x_scaled[:split_point]
y_train = y[:split_point]
X_test = x_scaled[split_point:]
y_test = y[split_point:]

time_step = 3
Xtrain = []
ytrain = []
Xtest = []
ytest = []
for i in range(time_step,len(X_train)):
  Xtrain.append(X_train[i-time_step:i,:X_train.shape[1]]) # we want to use the last 3 days’ data to predict the next day
  ytrain.append(y_train[i])
for i in range(time_step, len(y_test)):
  Xtest.append(X_test[i-time_step:i,:X_test.shape[1]])
  ytest.append(y_test[i])


Xtrain, ytrain = np.array(Xtrain), np.array(ytrain)
Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2]))
Xtest, ytest = np.array(Xtest), np.array(ytest)
Xtest = np.reshape(Xtest,(Xtest.shape[0],Xtest.shape[1],Xtest.shape[2]))


model = load_model('keras_model.h5')
model_load_status = st.text("loading data.")
model_load_status = st.text("loading data..")
model_load_status = st.text("loading data...")


# Prediction phase
train_predict = model.predict(Xtrain)
test_predict = model.predict(Xtest)

# Concatenating with an array of zeros tho allow for proper input structure
train_predict = np.c_[train_predict,np.zeros(train_predict.shape)]
test_predict = np.c_[test_predict,np.zeros(test_predict.shape)]


# Doing an inverse_transform to get the actual values
train_predict = scaler.inverse_transform(train_predict)
train_predict = [x[0] for x in train_predict]

test_predict = scaler.inverse_transform(test_predict)
test_predict = [x[0] for x in test_predict]

# Instantiating the original price values
original_stock_price = [y for y in x[split_point:][price_of_choice]]


def plot_combined_figures():
    layout = go.Layout(showlegend=True)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=data.index, y=original_stock_price, name="Actual stock price trajectory"))
    fig.add_trace(go.Scatter(x=data.index, y=test_predict, name="Predicted stock price trajectory"))
    fig.layout.update(title_text="Combined plot for actual and predicted values", xaxis_rangeslider_visible=True)
    fig.update_layout(
        width=1000,
        height=650
    )
    st.plotly_chart(fig)

plot_combined_figures()