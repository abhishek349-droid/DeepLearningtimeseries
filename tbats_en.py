
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re

train = pd.read_csv('/content/drive/MyDrive/web-traffic-time-series-forecasting/train_1.csv').fillna(0)
print(train.head())

print(train.info())

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

print(total_view['en'])

plot.plot(days,total_view['en'],label =labels['en'])
plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Total views of wikipedia pages in Japanese Language")
plot.show()

import math

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

info = pd.DataFrame(total_view['en'])

info.to_csv('/content/drive/MyDrive/tbats_en.csv')

from pandas import read_csv
dataframe = read_csv('/content/drive/MyDrive/nnar_en.csv', usecols=[1])

dataset = dataframe.values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train):len(dataset), :] = test

import matplotlib.pyplot as plt

#plt.plot(scaler.inverse_transform(dataset),label='Original',color="black")
plt.plot(train,label='Train',color="blue")
plt.plot(testPredictPlot,label="Test",color="#57A62C")
plot.title('Expected Vs Predicted Views Forecasting of total view of English articles')
plot.xlabel('Days')
plot.ylabel('Total Views')
plt.legend()
plt.show()

!pip install tbats
from tbats import TBATS, BATS

estimator = TBATS(seasonal_periods = (7,len(train)))
model = estimator.fit(train)
#MAPE(tbats_forecast$mean, validation) * 100

# Forecast ahead
y_forecasted = model.forecast(steps=len(test))
    
# Summarize fitted model
#print(y_forecasted)

#print(test)

y_forecasted=y_forecasted.reshape(y_forecasted.shape[0],1)
yPlot = np.empty_like(dataset)
yPlot[:, :] = np.nan
yPlot[len(train):len(dataset), :] = y_forecasted

import matplotlib.pyplot as plt

plt.plot((dataset),label='Original',color="black")
plt.plot(train,label='Train',color="blue")
#plt.plot(testPredictPlot,label="Forecast",color="#57A62C")
plt.plot(yPlot,label="Forecast",color="#C91841")
plot.title('Expected Vs Predicted Views Forecasting of total view of English articles')
plot.xlabel('Days')
plot.ylabel('Total Views')
plt.legend()
plt.show()

from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(test,y_forecasted))
print(rms*100)

