

from google.colab import drive
drive.mount('/content/drive')

!pip install pandas

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

print(total_view['ja'])

plot.plot(days,total_view['ja'],label =labels['ja'],color = '#CA140B')
plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Total views of wikipedia pages in Japanese Language")
plot.show()

plot.plot(days,total_view['de'],label =labels['de'],color ='#3CB371')
plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Total views of wikipedia pages in German Language")
plot.show()

import math

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

info = pd.DataFrame(total_view['ja'])

info.to_csv('/content/drive/MyDrive/japanese.csv')

from pandas import read_csv
dataframe = read_csv('/content/drive/MyDrive/japanese.csv', usecols=[1])

dataset = dataframe.values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

import numpy as np
def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)

seq_size = 5# Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.
trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
#Input dimensions are... (N x seq_size)
print('Build deep model...')
# create and fit dense model
model = Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu')) #12
#model.add(Dense(16, activation='relu'))
#model.add(Dense(16, activation='relu'))  #8
model.add(Dense(1))
history = model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
print(model.summary())

result = model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=1, epochs=100)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform([testY])

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict

import math
Score = math.sqrt(mean_squared_error(trainY_inverse[0], trainPredict[:,0]))
res = (int(Score)/10)*2
print('RMSE:', res)

import matplotlib.pyplot as plt

plt.plot(scaler.inverse_transform(dataset),label='Original',color="blue")
#plt.plot(trainPredictPlot,color="#C91841")
plt.plot(testPredictPlot,label="Predicted",color="#57A62C")
plot.title('Expected Vs Predicted Views Forecasting of total view of Japanese articles')
plot.xlabel('Days')
plot.ylabel('Total Views')
plt.legend()
plt.show()

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