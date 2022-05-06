import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
import xgboost
import joblib
from datetime import datetime
from datetime import timedelta
from datetime import date
import yfinance as yf
import streamlit as st

st.set_page_config(
     page_title="Stock Predictor",
     page_icon=":stock:",
     layout="centered",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://finance.yahoo.com/',
         'Report a bug': None,
         'About': "# A simple web app to visualize stock data and predict stock closing prices"
     }
 )
st.title('Stock Trend Prediction')
# taking user input for stock ticker
stock_ticker = st.text_input('Enter the stock ticker', 'AAPL')
end_date = st.text_input('Enter end date, YYYY-MM-DD', str(date.today()))
start = pd.to_datetime(['2007-01-01']).astype(int)[0]//10**9 # convert to unix timestamp.
end = pd.to_datetime([end_date]).astype(int)[0]//10**9 # convert to unix timestamp.
url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_ticker + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'

# loading data
df = pd.read_csv(url)
df.dropna(inplace = True)

# get stock info
stock_info = yf.Ticker(stock_ticker)
info = stock_info.info['longBusinessSummary']
st.subheader('Company Information')
st.write(info)

# describe data
st.write('Stock data of last five days:')
st.write(df.tail())
st.write('Description of the data, showing mean, minimum, maximum, etc.:')
st.write(df.describe())

# displaying Adjusted Closing Price vs Time
st.subheader('Adjusted Closing Price vs. Time')
fig = plt.figure(figsize = (12, 6))
plt.plot(df['Adj Close'], label='Adjusted Closing Price')
plt.legend()
st.pyplot(fig)

# giving info about the moving average
st.write('Moving averages are a technical indicator for price movements. They are averages of the closing prices of stocks over varying durations-long term, mid-term to short term. For example, there can be a 200-day, 100-day, or 50-day moving average, based on what investors want to infer about the prices. A 100-day Moving Average (MA) is the average of closing prices of the previous 100 days or 20 weeks. It represents price trends over the mid-term.')

# plotting 100 days moving average
st.subheader('100 Days Moving Average')
ma100 = df['Adj Close'].rolling(100).mean() # 100 days moving average
fig = plt.figure(figsize = (12, 6))
plt.plot(df['Adj Close'], label='Closing Price')
plt.plot(ma100, label='100 Days MA')
plt.legend()
st.pyplot(fig)

# plotting 200 days moving average
st.subheader('200 Days Moving Average')
ma100 = df['Adj Close'].rolling(100).mean() # 100 days moving average
ma200 = df['Adj Close'].rolling(200).mean() # 200 days moving average
fig = plt.figure(figsize = (12, 6))
plt.plot(df['Adj Close'], label='Closing Price')
plt.plot(ma100, label='100 Days MA')
plt.plot(ma200, label='200 Days MA')
plt.legend()
st.pyplot(fig)

# preprocessing the data
data_training = pd.DataFrame(df['Adj Close'][:100])
data_testing = pd.DataFrame(df['Adj Close'][100:int(len(df))])
# scaling the data
sca = MinMaxScaler(feature_range=(0,1))
past100 = data_training.tail(100)
data_testing = past100.append(data_testing, ignore_index = True)
input_data = np.array(data_testing)
input_data = sca.fit_transform(input_data)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

# loading the trained model and predicting trends
model = joblib.load('ensemble_model.sav')
y_pred = model.predict(x_test)
st.subheader('Predicted Price vs. Actual Price')
fig = plt.figure(figsize=(12,6))
y_test = y_test.reshape(-1, 1)
y_pred = y_pred.reshape(-1, 1)
plt.plot(sca.inverse_transform(y_test), 'b', label='Closing Price')
plt.plot(sca.inverse_transform(y_pred), 'r', label='Predicted Closing Price')
plt.legend()
st.pyplot(fig)

# showing the latest prediction
dat = str(df['Date'].loc[len(df['Date'])-1])
Begindate = datetime.strptime(dat, "%Y-%m-%d")
Enddate = Begindate + timedelta(days=1)
fin_date = Enddate.strftime('%Y-%m-%d')
x1 = data_testing.tail(100)
x1 = sca.fit_transform(x1)
x1 = x1.reshape(1, 100)
y_pred1 = model.predict(x1)
d = np.array([y_pred1])
a = sca.inverse_transform(d)
stock_val = a[0, 0]
st.subheader('Latest Prediction')
st.write('The predicted Closing Price for ', stock_ticker, ' stock on ', fin_date, ' is ', stock_val)