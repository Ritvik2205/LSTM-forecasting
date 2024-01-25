import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import datetime


def str_to_datetime(s):
    split = s.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

def window_data(df, n=3):
    windowed_df = pd.DataFrame()
    windowed_df['Target Dates'] = df['Date']
    for i in range(n, 0, -1):
        windowed_df[f'Target-{i}'] = df['Close'].shift(i)   
    windowed_df['Target'] = df['Close']
    return windowed_df.dropna()

def windowed_df_to_date_X_y(windowed_df):
  df_as_np = windowed_df.to_numpy()

  dates = df_as_np[:, 0]

  mid_df = df_as_np[:, 1:-1]
  X = mid_df.reshape((len(dates), mid_df.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

df = pd.read_csv('MSFT.csv')
df = df[['Date', 'Close']]
df['Date'] = df['Date'].apply(str_to_datetime)
# df.index = df.pop('Date')
windowed_df = window_data(df)
dates, X, y = windowed_df_to_date_X_y(windowed_df)


q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]



plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])

plt.show()