#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


import matplotlib.pyplot as plt 


# In[7]:


import numpy as np
from sklearn.linear_model import LinearRegression 


# In[8]:


data = pd.read_excel("nasa.xlsx")
#data = data.ffill()
data.head


# In[9]:


# setting the 3 first columns to a proper date column 
data["date"] = pd.to_datetime(dict(year=data["YEAR"], month=data["MO"], day=data["DY"]))
date = data.set_index("date")



# In[10]:


date


# In[11]:


# drop 3 first columns and set the date as an index 
data = data.drop(columns = ["YEAR","MO","DY"])


# In[12]:


data = data.set_index("date")


# In[13]:


data.head(10)


# In[14]:


data = data.replace (-999,np.nan)
data.isna().sum().sum()


# In[15]:


data = data.interpolate()


# In[16]:


data.isna().sum()


# In[17]:


#view the radiation for a specific year
data.loc["2015","CLRSKY_SFC_SW_DWN"].plot(figsize=(12,5))
plt.title("Solar Radiation ")
plt.show()


# In[18]:


#view the solar radiation for all the time table 

data["CLRSKY_SFC_SW_DWN"].plot(figsize=(12,5))
plt.title("Solar Radiation")
plt.show()


# In[19]:


split =int( len(data)*0.8)
# from row 1 to split  which is 7199 will be the train dataset 
train = data[:split]
test = data[split:]
#train = 80% of data , test  = 20% of data


# In[20]:


#checking for outliers  since we will scale data if we have outliers it will have a lot of impact on it 
train["ALLSKY_SFC_SW_DWN"].describe()


# In[21]:


#another box plot to be sure 
train["ALLSKY_SFC_SW_DWN"].plot.box()
# decision , we do not have outliers ! 


# In[23]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# choose input features
features = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DIFF",
    "ALLSKY_KT",
    "WS10M",
    "RH2M"
]

# convert train and test to arrays
train_series = train[features].values
test_series = test[features].values

# scale inputs
scaler_x = MinMaxScaler()
train_scaled = scaler_x.fit_transform(train_series)
test_scaled = scaler_x.transform(test_series)

# scale target separately
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(train[["ALLSKY_SFC_SW_DWN"]].values)
y_test_scaled = scaler_y.transform(test[["ALLSKY_SFC_SW_DWN"]].values)

# create sequences
def create_sequences(X_data, y_data, window):
    X, y = [], []
    for i in range(len(X_data) - window):
        X.append(X_data[i:i+window])
        y.append(y_data[i+window])
    return np.array(X), np.array(y)

# use past 30 days to predict next day radiation
window = 30
X_train, y_train = create_sequences(train_scaled, y_train_scaled, window)
X_test, y_test = create_sequences(test_scaled, y_test_scaled, window)

# check shapes
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# build LSTM model
model = Sequential()
model.add(LSTM(128, activation="tanh", input_shape=(window, len(features))))
model.add(Dense(1))

# compile model
model.compile(optimizer="adam", loss="mse")

# train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# predict
pred_scaled = model.predict(X_test)

# inverse transform predictions
pred = scaler_y.inverse_transform(pred_scaled)
y_test_real = scaler_y.inverse_transform(y_test)

# plot
plt.figure(figsize=(12,5))
plt.plot(test.index[window:], y_test_real.ravel(), label="Real")
plt.plot(test.index[window:], pred.ravel(), label="Predicted")
plt.title("LSTM Forecast of Solar Radiation")
plt.legend()
plt.show()


# In[24]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# keep only target variable
train_series = train["ALLSKY_SFC_SW_DWN"].values.reshape(-1, 1)
test_series = test["ALLSKY_SFC_SW_DWN"].values.reshape(-1, 1)

# scale data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_series)
test_scaled = scaler.transform(test_series)

# create sequences
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# use past 30 days to predict next day
window = 30
X_train, y_train = create_sequences(train_scaled, window)
X_test, y_test = create_sequences(test_scaled, window)

# build RNN
model = Sequential()
model.add(SimpleRNN(128, activation="tanh", input_shape=(window, 1)))
model.add(Dense(1))

# compile model
model.compile(optimizer="adam", loss="mse")

# train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# predict
pred_scaled = model.predict(X_test)

# inverse transform
pred = scaler.inverse_transform(pred_scaled)
y_test_real = scaler.inverse_transform(y_test)

# plot
plt.figure(figsize=(12,5))
plt.plot(test.index[window:], y_test_real, label="Real")
plt.plot(test.index[window:], pred, label="Predicted")
plt.title("RNN Forecast of Solar Radiation 30 day trainging ")
plt.legend()
plt.show()


# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test_real.ravel(), pred.ravel())
rmse = np.sqrt(mean_squared_error(y_test_real.ravel(), pred.ravel()))

print("MAE:", mae)
print("RMSE:", rmse)


# In[ ]:





# In[ ]:




