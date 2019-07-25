
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import urllib
import sys
import redis
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

# Global variables and settings
plt.style.use('ggplot')
np.random.seed(233)
TRAINING_MODE = True
SPLIT_RATIO = 0.85
RESAMPLE_INTERVAL = '1.5T'
Export_Data_Path = 'Output_Data/'
Export_Graph_Path = 'Graphic_Aid/'

class IO:
	'''Input and output module'''
	def data_import(resample = True, resample_interval = '2T'):
		'''Import JSON formatted data'''
		try:
			df_raw = pd.read_json('https://rice.cn1.utools.club/data/test/average')
			df_raw.to_json('average.json')
		except urllib.error.HTTPError:
			df_raw = pd.read_json('average.json')
		df = df_raw[['t1', 't2', 't3', 'l1', 'l2', 'l3', 'u1', 'u2', 'u3']]
		index = df_raw['operateTime']
		df.set_index(pd.to_datetime(index, unit = 'ms'), inplace = True)
		df = df.astype('float32')
		if resample:
			df = df.resample(resample_interval).last().ffill()
		index = df.index
		columns = df.columns
		return df, index, columns

class Visualisation:
	'''Create intuitive plots'''
	def exploratory_plot_t1(df):
		# Visual reference for correlation
		plt.figure()
		pd.plotting.lag_plot(df['t1'])
		plt.savefig(Export_Graph_Path + 'lag_plot_t1.png', dpi = 500)
		plt.close()
		# Autocorrelation plot
		plt.figure()
		pd.plotting.autocorrelation_plot(df['t1'])
		plt.savefig(Export_Graph_Path + 'autocorrelation_plot_t1', dpi = 500)
		plt.close()

	def all_factor_plot(df):
		values = df.values
		scaler = MinMaxScaler(feature_range = (0,1))
		df = pd.DataFrame(scaler.fit_transform(values), columns = df.columns, index = df.index)
		plt.figure(figsize = (40, 3))
		plt.plot(df)
		plt.legend(df.columns)
		plt.savefig(Export_Graph_Path + 'all_data_plot.svg', dpi = 500)
		plt.close()
		plt.figure(figsize = (10, 8))
		i = 1
		for vector in range(len(df.columns)):
			plt.subplot(len(df.columns), 1, i)
			plt.plot(values[:, vector])
			plt.title(df.columns[vector], y = 0.5, loc = 'right')
			i += 1
		plt.savefig(Export_Graph_Path + 'all_data_decomposition_plot.png', dpi = 500)
		plt.close()

	def autoregression_result_plot(df):
		plt.figure()
		df.plot()
		plt.savefig('autoregression_prediction_plot.png', dpi = 500)
		plt.close()
	
	def rnn_history_plot(history):
		plt.figure()
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc = 'upper left')
		plt.savefig(Export_Graph_Path + 'model_accuracy_plot.png', dpi = 500)
		plt.close()

	def rnn_result_plot(values, train_predict_plot, test_predict_plot, index):
		df = pd.DataFrame({'Actual':values, 'train_predict':train_predict_plot, 'test_predict':test_predict_plot})
		df.to_csv('test_file.csv')
		plt.figure(dpi = 500, figsize = (10,6))
		plt.title('Prediction Plot Based on Recurrent Neural Network')
		plt.plot(df)
		plt.xlabel('Range Index for Input')
		plt.ylabel('Associated Parameter')
		plt.legend(['Actual', 'Train Predicted', 'Test Predicted'])
		plt.savefig('rnn_preiction_plot.png')
		plt.close()

class AR:
	'''Perform analytics on importe data'''
	# Deprecated
	def to_stationary(df):
		'''Remove seasonality and trend brutally'''
		df = df.diff()
		return df

	def train_test_split_stats(df):
		train, test = np.split(df, [int(0.8 * len(df.array))])
		return train,test
	
	def autoregression_stats(train, test, index):
		model = AR(train)
		fitted = model.fit()
		predictions = pd.DataFrame(fitted.predict(start = len(train), end = len(train) + len(test) - 1, dynamic = False))
		predictions.set_index(test.index, inplace = True)
		predictions.columns = ['predicted']
		return predictions

class RNN:		
	def execution(df, training_mode, n_steps_in = 100, n_steps_out = 20, split_ratio = 0.8):
		# Normaliza data
		X, y, z, test, n_features, train_scaler, test_scaler = Training.normalization(df, n_steps_in, n_steps_out, split_ratio)

		# Train or import model architecture, weights and configurations
		if training_mode:
			model, history = Training.architecture(X, y, n_steps_in, n_steps_out, n_features)
			Visualisation.rnn_history_plot(history)
		else:
			model = load_model('rnn_model.h5')

		# Visualize model structure
		print(model.summary())
		plot_model(model, show_shapes = True, to_file = Export_Graph_Path + 'rnn_model.png')

		# Evaluate model accuracy
		x_test, y_test = Utilities.split_sequences(test, n_steps_in, n_steps_out)
		model.evaluate(x = x_test, y = y_test, callbacks = [EarlyStopping(monitor = 'loss')])
		predicted = model.predict(x_test)

		# Monitor dimensionality of output and test data
		print('\nThe dimensionality of test data is: ' + str(y_test.shape))
		print('The dimensionality of predicted data is: ' + str(predicted.shape) + '\n')

		# A sanity check on actual data
		np.set_printoptions(precision = 1, suppress = True)
		print('**********Actual ' + str(n_steps_out) + ' Tail Data**********')
		print(test_scaler.inverse_transform(y_test[-1]))
		print('**********End**********\n')
		print('**********Predicted ' + str(n_steps_out) + ' Tail Data**********')
		print(test_scaler.inverse_transform(predicted[-1]))
		print('**********End**********\n')
		
		# Return a model, a scaler, a parameter and the data for model deployment
		return model, train_scaler, test_scaler, n_steps_in, n_steps_out, z

	def deployment(z, model, train_scaler, test_scaler, n_steps_in, n_steps_out, columns, index, freq):
		predicted = test_scaler.inverse_transform(model.predict(z)[0])
		future_index = pd.date_range(start = index.array[-1], periods = n_steps_out + 1, freq = freq, closed = 'right')
		df = pd.DataFrame(predicted, columns = columns, index = future_index)
		print('\n**********Predicted Future ' + str(n_steps_out) + ' Tail Data with ' + str(n_steps_in) + ' Lookback**********')
		print(df)
		print('**********End**********\n')
		df.to_json(Export_Data_Path + 'Predicted_Future_Values.json')

class Training:
	def normalization(df, n_steps_in, n_steps_out, split_ratio):
		values = df.values
		train_scaler = MinMaxScaler(feature_range = (0,1))
		test_scaler = MinMaxScaler(feature_range = (0,1))
		train, test = Utilities.train_test_split(values, split_ratio)
		train, test = train_scaler.fit_transform(train), test_scaler.fit_transform(test)
		z = np.array([test[-n_steps_in:]])
		X, y = Utilities.split_sequences(train, n_steps_in, n_steps_out)
		n_features = X.shape[2]
		X = X.reshape((X.shape[0], X.shape[1], n_features))
		y = y.reshape((y.shape[0], y.shape[1], n_features))
		z = z.reshape((z.shape[0], z.shape[1], n_features))
		return X, y, z, test, n_features, train_scaler, test_scaler

	def architecture(X, y, n_steps_in, n_steps_out, n_features):
		'''Define model architecture'''
		inputs = Input(shape = (n_steps_in, n_features))
		encoder = Bidirectional(LSTM(96, activation = 'relu', return_sequences = False, dropout = 0.3))(inputs)
		multistep = RepeatVector(n_steps_out)(encoder)
		decoder = Bidirectional(LSTM(96, activation = 'relu', return_sequences = True, dropout = 0.3))(multistep)
		output = TimeDistributed(Dense(n_features))(decoder)
		model = Model(inputs, output)
		model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
		history = model.fit(X, y, validation_split = 0.15, epochs = 150, verbose = 1, callbacks = [EarlyStopping(monitor = 'loss')])
		model.save('rnn_model.h5')
		return model, history

class Utilities:
	'''Auxillary functions'''
	def dataframe_for_plot(df, predictions):
		df_plot = pd.concat([df, predictions], axis = 1)
		df_plot.columns = ['tf_actual', 'tf_predicted']
		return df_plot

	def create_dataset(df_values, look_back = 1):
		'''For the LSTM Recurrent Neural Network'''
		X, Y = [], []
		for i in range(len(df_values) - look_back - 1):
			a = df_values[i:(i + look_back), 0]
			X.append(a)
			Y.append(df_values[i + look_back, 0])
		return np.array(X), np.array(Y)
	
	def split_sequences(sequences, n_steps_in, n_steps_out):
		X, y = list(), list()
		for i in range(len(sequences)):
			end_ix = i + n_steps_in
			out_end_ix = end_ix + n_steps_out
			if out_end_ix > len(sequences):
				break
			seq_x, seq_y = sequences[i:end_ix], sequences[end_ix: out_end_ix]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)

	def train_test_split(df_values, split_ratio):
		'''Split data into training and testing sets'''
		train_size = int(len(df_values) * split_ratio)
		test_size = len(df_values) - train_size
		train, test = df_values[0:train_size, :], df_values[train_size:len(df_values), :]
		print('\nTrain data size: ' + str(train_size) + '\n' + 'Test data size: ' + str(test_size) + '\n')
		return train, test


def main():
	df, index, columns = IO.data_import(resample = True, resample_interval = RESAMPLE_INTERVAL)
	Visualisation.exploratory_plot_t1(df)
	Visualisation.all_factor_plot(df)
	model, train_scaler, test_scaler, n_steps_in, n_steps_out, z = RNN.execution(df, training_mode = TRAINING_MODE, split_ratio = SPLIT_RATIO)

	print('\n**********Actual Tail Data For Training**********')
	print(df[- n_steps_in - n_steps_out:- n_steps_out])
	print('**********End**********\n')

	RNN.deployment(z, model, train_scaler, test_scaler, n_steps_in, n_steps_out, columns, index, freq = RESAMPLE_INTERVAL)




if __name__ == '__main__':
	main()
