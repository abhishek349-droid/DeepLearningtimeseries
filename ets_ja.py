

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re

train = pd.read_csv('/content/drive/MyDrive/time_series_dataset/train_1.csv').fillna(0)
print(train.head())

def lang(Page):
    val = re.search('[a-z][a-z].wikipedia.org',Page)
    if val:
        #print(val)
        return val[0][0:2] 
                 
    
    # no_lang for media files ; wikimedia.org
    return 'no_lang'

train['language'] = train.Page.map(lang)

# Article Count 
print("\nArticle count as per Language : \n", Counter(train.language))

language_set = {}
language_set['en'] = train[train.language=='en'].iloc[:,0:-1]
language_set['ja'] = train[train.language == 'ja'].iloc[:, 0:-1]
language_set['de'] = train[train.language == 'de'].iloc[:, 0:-1]
language_set['fr'] = train[train.language == 'fr'].iloc[:, 0:-1]
language_set['ru'] = train[train.language == 'ru'].iloc[:, 0:-1]
language_set['es'] = train[train.language == 'es'].iloc[:, 0:-1]
language_set['no_lang'] = train[train.language == 'no_lang'].iloc[:, 0:-1]

print(language_set['no_lang'])
print( train[train.language=='en'].iloc[:,0:-1])

for key in language_set:
    print("KEY : ", language_set[key],"\n")

# axis =0 : vertical in NumPy ;   axis =1 : horizontal in NumPy
total_view = {} 
for key in language_set:
    total_view[key] = language_set[key].iloc[:, 1:].sum(axis=0) / language_set[key].shape[0]


for key in language_set:
    print("KEY : ", key)
    print("\nTotal_Value KEY : \n", total_view[key])

#print(total_view['en'].shape[0])
days =[r
       for r in range(total_view['en'].shape[0])]
#total views graph
plot.figure(figsize=(9,6))
labels={'ja':'Japanese','de':'German','en' : 'English','no_lang':'Media_File','fr':'French','zh':'Chinese',
        'ru':'Russian','es':'Spanish'}

for key in total_view:
   plot.plot(days,total_view[key],label = labels[key])

plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Language Influences Page Total_View")
plot.legend(loc= 'upper left',bbox_to_anchor= (1.1,1))
plot.show()

info = pd.DataFrame(total_view['ja'])

info.to_csv('/content/drive/MyDrive/japanese.csv')

from pandas import read_csv
dataframe = read_csv('/content/drive/MyDrive/japanese - japanese.csv', usecols=[1])

dataset = dataframe.values

# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print(dataframe)

dataframe.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
m = 12
alpha = 1/(2*m)

dataframe['HWES1'] = SimpleExpSmoothing(dataframe["Total Views"]).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
dataframe[['Total Views','HWES1']].plot(title='Holt Winters Single Exponential Smoothing for Japanese')
plot.xlabel('Days')
plot.ylabel('Total Views')

dataframe['HWES2_ADD'] = ExponentialSmoothing(dataframe['Total Views'],trend='add').fit().fittedvalues
dataframe['HWES2_MUL'] = ExponentialSmoothing(dataframe['Total Views'],trend='mul').fit().fittedvalues
dataframe[['Total Views','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend for japanese');
plot.xlabel('Days')
plot.ylabel('Total Views')

dataframe['HWES3_ADD'] = ExponentialSmoothing(dataframe['Total Views'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
dataframe['HWES3_MUL'] = ExponentialSmoothing(dataframe['Total Views'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues
dataframe[['Total Views','HWES3_ADD','HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality for Japanese')
plot.xlabel('Days')
plot.ylabel('Total Views')

# Split into train and test set
Original = dataframe[:550]
train_ets = dataframe[:275]
test_ets = dataframe[275:]

fitted_model = ExponentialSmoothing(train_ets['Total Views'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(50)
train_ets['Total Views'].plot(legend=True,label='TRAIN')
test_ets['Total Views'].plot(legend=True,label='Predicted',figsize=(6,4))
test_ets.plot(legend=True,label='PREDICTION')
Original['Total Views'].plot(legend =True,label='Original')
plt.title('Train, Test and Predicted Test using Holt Winters for Japanese')
plot.xlabel('Days')
plot.ylabel('Total Views')

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

def mean_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) / y_true * 100)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def running_diff(arr, N):
    return np.array([arr[i] - arr[i-N] for i in range(N, len(arr))])

def mean_absolute_scaled_error(training_series, testing_series, prediction_series):
    errors_mean = np.abs(testing_series - prediction_series ).mean()
    d = np.abs(running_diff(training_series, 12) ).mean()
    return errors_mean/d

from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
MAE = mean_absolute_error(test_ets,train_ets)
MSE = mean_squared_error(test_ets,train_ets)
RMSE = sqrt(MSE)/10
print('RMSE:',RMSE)

npages = 5
top_pages = {}
key = 'ja'
print(key)
sum_set = pd.DataFrame(language_set[key][['Page']])
sum_set['total'] = language_set[key].sum(axis=1)
sum_set = sum_set.sort_values('total',ascending=False)
print(sum_set.head(5))
top_pages[key] = sum_set.index[0]
print('\n')