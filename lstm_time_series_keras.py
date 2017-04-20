
# coding: utf-8

# In[1]:

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout


# In[2]:

d = pandas.read_csv('stages.csv')


# In[3]:

dataset = d.values
dataset = dataset.astype('float32')


# In[4]:

plt.figure(figsize = (14,8))
plt.plot(dataset)
plt.show()


# In[5]:

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[6]:

# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[7]:

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[8]:

#look_back -> od koliko prethodnih dana pravi sekvencu
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[ ]:


trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[ ]:

model = Sequential()
model.add(LSTM(5, input_dim=look_back))
#sa dropoutom malo sporije radi i sa 20 % rezultat nije bolji
#model.add(Dropout(0.2))
model.add(Dense(1))
#adam optimizer daje najbolje rezultate
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=6, batch_size=1, verbose=2)


# In[ ]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
testPredict.shape


# In[ ]:

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[ ]:



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[ ]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize = (14,8))

'''
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''
plt.plot(scaler.inverse_transform(dataset[3755:4015]))
plt.plot(trainPredictPlot[3755:4015])
plt.plot(testPredictPlot[3755:4015])
plt.show()



# In[ ]:

d.shape


# In[ ]:




# In[ ]:



