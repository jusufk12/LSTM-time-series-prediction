import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from lstm1 import generate_data, lstm_model

LOG_DIR = './ops_logs/sin'
TIMESTEPS = 5
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 128

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                            model_dir=LOG_DIR)
data = pd.read_csv('stages.csv')

X, y = generate_data(data, TIMESTEPS, seperate=False)


regressor.fit(X['train'], y['train'], 
              batch_size=BATCH_SIZE,
              steps=TRAINING_STEPS)


predicted_Train = regressor.predict(X['train'])
plot_predicted_train, = plt.plot(predicted_Train, label='predicted')
plot_test, = plt.plot(y['train'], label='test')
plt.legend(handles=[plot_predicted_train, plot_test])
plt.show()


predicted = regressor.predict(X['test'])
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
