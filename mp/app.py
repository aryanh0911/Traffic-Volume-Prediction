from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn import svm
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns  # 基于matplolib的画图模块
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def posix_time(dt):
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)


data = pd.read_csv('static/Train.csv')
data = data.sort_values(
    by=['date_time'], ascending=True).reset_index(drop=True)
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)
data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
data.to_csv("traffic_volume_data.csv", index=None)
# data.columns
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
data = pd.read_csv("traffic_volume_data.csv")
# Use all available data or sample up to 10000 if more records exist
sample_size = min(len(data), 10000)
data = data.sample(sample_size).reset_index(drop=True)
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday',  'temperature',
                   'weekday', 'hour', 'month_day', 'year', 'month']
# ohe_encoder = OneHotEncoder()
# x_ohehot = ohe_encoder.fit_transform(data[label_columns])
# ohe_features = ohe_encoder.get_feature_names()
# x_ohehot = pd.DataFrame(x_ohehot.toarray(),
#                         columns=ohe_features)
# data = pd.concat(
#     [data[['date_time']], data[['traffic_volume']+numeric_columns], x_ohehot], axis=1)
# data['traffic_volume'].hist(bins=20)
# metrics = ['month', 'month_day', 'weekday', 'hour']

# fig = plt.figure(figsize=(8, 4*len(metrics)))
# for i, metric in enumerate(metrics):
# 	ax = fig.add_subplot(len(metrics), 1, i+1)
# 	ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
# 	ax.set_xlabel(metric)
# 	ax.set_ylabel("Mean Traffic")
# 	ax.set_title(f"Traffic Trend by {metric}")
# plt.tight_layout()
# plt.show()

#features = numeric_columns+list(ohe_features)
features = numeric_columns
target = ['traffic_volume']
X = data[features]
y = data[target]
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()
warnings.filterwarnings('ignore')
##################
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
print("Model training completed!")
print("Sample predictions:", regr.predict(X[:10]))
print("Sample actual values:", y[:10])
########################################################################################################
app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('index.html')


d = {}


@app.route('/predict', methods=['POST'])
def predict():
    d['is_holiday'] = request.form['isholiday']
    if d['is_holiday'] == 'yes':
        d['is_holiday'] = int(1)
    else:
        d['is_holiday'] = int(0)
    d['temperature'] = int(request.form['temperature'])
    d['weekday'] = int(0)
    D = request.form['date']
    d['hour'] = int(request.form['time'][:2])
    d['month_day'] = int(D[8:])
    d['year'] = int(D[:4])
   # should change
    d['month'] = int(D[5:7])
    # Since the model was trained only on numeric features, we'll use only those
    # The weather data will be ignored for now to match the training
    final = []
    final.append(d['is_holiday'])
    final.append(d['temperature'])
    final.append(d['weekday'])
    final.append(d['hour'])
    final.append(d['month_day'])
    final.append(d['year'])
    final.append(d['month'])
    print(f"Input features: {final}")
    print(f"Number of features: {len(final)}")
    prediction = regr.predict([final])[0]
    # Scale back the prediction to original range
    prediction_scaled = y_scaler.inverse_transform([[prediction]])[0][0]
    print(f"Predicted traffic volume: {prediction_scaled}")
    # Store weather info for display
    d['weather_type'] = request.form.get('x0')
    d['weather_description'] = request.form.get('x1')
    return render_template('output.html', data1=d, data2=final, prediction=prediction_scaled)
if __name__ == '__main__':
    app.run(debug=True)
