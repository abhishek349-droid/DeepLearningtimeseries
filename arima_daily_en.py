

from google.colab import drive
drive.mount('/content/drive')

!pip install pandas

import pandas as pd
import numpy as dragon
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re

train = pd.read_csv('/content/drive/MyDrive/time_series_dataset/train_1.csv.zip').fillna(0)
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
    total_view[key] = language_set[key].iloc[:, 1:].sum(axis=0) / language_set[key].shape[0] #Total views per day


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

plot.plot(days,total_view['en'],label =labels['en'])
plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Total views of wikipedia pages in English Language")
plot.show()

plot.plot(days,total_view['de'],label =labels['de'],color ='#3CB371')
plot.ylabel("Views per Page")
plot.xlabel("Days (01-07-2015 to 31-12-2016)")
plot.title("Total views of wikipedia pages in German Language")
plot.show()

!pip install statsmodels

from statsmodels.tsa.stattools import adfuller

def test_stationarity(x):
  #calculating rolling mean and std
  rolmean = x.rolling(window=22,center=False).mean()
  rolstd = x.rolling(window=22,center=False).std()
  # Plotting 
  original = plot.plot(x.values,color='#C70039',label='Original')
  mean = plot.plot(rolmean.values,color ='#FFC300',label='Rolling Mean')
  std = plot.plot(rolstd.values,color ='#0F9FC5',label='Rolling Std')

  plot.legend(loc='best')
  plot.title("Original vs Rolling Mean & Standard Deviation")
  plot.show(block=False)

  result = adfuller(x)# dickey fuller test
  print('ADF statistic: %f' %result[0])
  print('p-value: %f' %result[1])
  pvalue = result[1]
  for key,value in result[4].items():
    if result[0] > value:
      print("The Graph is Non-Stationery")
      break
    else:
      print("The Graph is Stationery")
      break
  print('Critical Values:')
  for key,value in result[4].items():
    print('\t%s: %.3f' %(key,value))
  
test_stationarity(total_view['en'])

ts_log = dragon.log(total_view['en'])
plot.plot(ts_log.values,color="#900C3F")
plot.show()

test_stationarity(ts_log)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log.values,model='multiplicative',freq=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plot.subplot(411)
plot.title("Observerd = Trend + Seasonality + Residuals")
plot.plot(ts_log.values,label='Observed',color = '#6F3DD0')

plot.legend(loc='best')
plot.subplot(412)
plot.plot(trend,label='Trend',color='#6F3DD0')
plot.legend(loc='best')

plot.subplot(413)
plot.plot(trend,label='Seasonality',color='#6F3DD0')
plot.legend(loc='best')

plot.subplot(414)
plot.plot(trend,label='Residuals',color='#6F3DD0')
plot.legend(loc='best')
plot.tight_layout()
plot.show()

ts_log_decompose = residual

#Remove trend and seasonality with differencing

ts_log_diff = ts_log - ts_log.shift()
plot.plot(ts_log_diff.values,color = '#079894')
plot.show()

ts_log_diff.dropna(inplace =True)
test_stationarity(ts_log_diff)

"""Now the ADF statistic is much lesser than critical value at 1%.Therefore the graph is now stationery with 99% confidence interval."""

#ACF and PACF plots
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA

lag_acf = acf(ts_log_diff, nlags=10)
lag_pacf = pacf(ts_log_diff, nlags=10,method ='ols')

#plot ACF
plot.subplot(1,1,1)
plot.plot(lag_acf,color='#94075C')
plot.axhline(y=0,linestyle='--',color='#BFC412')
plot.title('Autocorrelation Function-ACF')
plot.show()

#plot PACF
plot.subplot(1,1,1)
plot.plot(lag_pacf,color='#94075C')
plot.axhline(y=0,linestyle='--',color='#BFC412')
plot.title('Partial Autocorrelation Function-PACF')
plot.tight_layout()
plot.show()

#AR model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
model = ARIMA(ts_log.values,order=(2,1,0))
results_arima = model.fit(disp=-1)
plot.plot(ts_log_diff.values,color='#088DA6')
plot.plot(results_arima.fittedvalues,color='#581845')
MAE = mean_absolute_error(results_arima.fittedvalues,ts_log_diff.values)
RSS = sum((results_arima.fittedvalues - ts_log_diff.values)**2)
MSE = mean_squared_error(results_arima.fittedvalues,ts_log_diff.values)
RMSE = sqrt(MSE)
print('RSS:',RSS,'MSE:',MSE,'RMSE:',RMSE,'MAE:',MAE)
#print('MSE:',MSE)

model = ARIMA(ts_log.values,order=(0,1,1))
results_arima = model.fit(disp=-1)
plot.plot(ts_log_diff.values,color='#088DA6')
plot.plot(results_arima.fittedvalues,color='#581845')
MAE = mean_absolute_error(results_arima.fittedvalues,ts_log_diff.values)
RSS = sum((results_arima.fittedvalues - ts_log_diff.values)**2)
MSE = mean_squared_error(results_arima.fittedvalues,ts_log_diff.values)
RMSE = sqrt(MSE)
print('RSS:',RSS,'MSE:',MSE,'RMSE:',RMSE,'MAE:',MAE)
#print('MSE:',MSE)

model = ARIMA(ts_log.values,order=(2,1,1))
results_arima = model.fit(disp=-1)
plot.plot(ts_log_diff.values,color='#088DA6')
plot.plot(results_arima.fittedvalues,color='#581845')
MAE = mean_absolute_error(results_arima.fittedvalues,ts_log_diff.values)
RSS = sum((results_arima.fittedvalues - ts_log_diff.values)**2)
MSE = mean_squared_error(results_arima.fittedvalues,ts_log_diff.values)
RMSE = sqrt(MSE)
print('RSS:',RSS,'MSE:',MSE,'RMSE:',RMSE,'MAE:',MAE)
#print('MSE:',MSE)

from statsmodels.tsa.arima_model import ARIMA

size = int(len(ts_log)-100)
train_arima,test_arima = ts_log[0:size],ts_log[size:len(ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()
print('Predicted vs Expected Values..')
print('\n')
for t in range(len(test_arima)):
  model = ARIMA(history,order=(2,1,1))
  model_fit = model.fit(disp=0)
  
  output = model_fit.forecast()
  pred_value = output[0]

#RSS = sum((- pred_value)**2)


  original_value = test_arima[t]
  history.append(original_value)
  pred_value = dragon.exp(pred_value)
  original_value = dragon.exp(original_value)

  
  error =((abs(pred_value - original_value))/original_value)*100
  error_list.append(error)
  print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')
    
  predictions.append(float(pred_value))
  originals.append(float(original_value))
    
    
print('\n Means Error in Predicting Test Case Articles : %f ' % (sum(error_list)/float(len(error_list))), '%')

plot.figure(figsize=(8, 6))
test_day = [t+450
           for t in range(len(test_arima))]
labels={'Orginal','Predicted'}
plot.plot(test_day, predictions, color= '#DFFF00')
plot.plot(test_day, originals, color = '#1F618D')
plot.title('Expected Vs Predicted Views of test case Articles')
plot.xlabel('Days')
plot.ylabel('Total Views')
plot.legend(labels)
plot.show()

plot.figure(figsize=(8, 6))
test_day = [t+450
           for t in range(len(test_arima))]
plot.plot(test_day, predictions, color= 'blue',label ='Original')
plot.plot(days, total_view['en'], color ='#DF025D',label ='Predicted')
plot.title('Expected Vs Predicted Views Forecasting of total view of English articles')
plot.xlabel('Days')
plot.ylabel('Total Views')
plot.legend()
plot.show()

#DF025D

npages = 5
top_pages = {}
key = 'en'
print(key)
sum_set = pd.DataFrame(language_set[key][['Page']])
sum_set['total'] = language_set[key].sum(axis=1)
sum_set = sum_set.sort_values('total',ascending=False)
print(sum_set.head(5))
top_pages[key] = sum_set.index[0]
print('\n')

def plot_entry(key,idx):
    data = language_set[key].iloc[idx,1:]
    fig = plot.figure(1,figsize=(10,5))
    plot.plot(days,data,color='#C91841')
    plot.xlabel('day')
    plot.ylabel('views')
    plot.title(train.iloc[language_set[key].index[idx],0])
    
    plot.show()
    
idx = [1, 2, 3, 4, 5]
for i in idx:
    plot_entry('en',i)