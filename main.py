import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# converting dates to datetime type
def str_to_datetime(s):
    split = s.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

# building dataframe with 3 day shifts
def window_data(df, n=3):
    windowed_df = pd.DataFrame()
    df_sliced = df.query("Date >= datetime.datetime(2021,3,25)")
    windowed_df['Target Dates'] = df_sliced['Date']
    for i in range(n, 0, -1):
        windowed_df[f'Target-{i}'] = df['Close'].shift(i)   
    windowed_df['Target'] = df['Close']
    return windowed_df.dropna()

# separating dates, target values, and previous 3-day values 
def windowed_df_to_date_X_y(windowed_df):
  df_as_np = windowed_df.to_numpy()

  dates = df_as_np[:, 0]

  mid_df = df_as_np[:, 1:-1]
  X = mid_df.reshape((len(dates), mid_df.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

# reading CSV file
df = pd.read_csv('MSFT.csv')
df = df[['Date', 'Close']]
df['Date'] = df['Date'].apply(str_to_datetime)
windowed_df = window_data(df)
dates, X, y = windowed_df_to_date_X_y(windowed_df)

# constants for separating training data 
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

# separating training data and test data
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# Building LSTM mpodel
model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

# Fitting training data
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Running the model
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

# Plotting the data 
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions', 
            'Training Observations',
            'Validation Predictions', 
            'Validation Observations',
            'Testing Predictions', 
            'Testing Observations'])

plt.show()

