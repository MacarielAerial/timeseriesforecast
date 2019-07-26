
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

# Graphic module setting and psudo-random initialisation
plt.style.use('ggplot')
np.random.seed(233)

# Global variables for engineers to adjust
Export_Data_Path = 'Output_Data/'
Export_Graph_Path = 'Graphic_Aid/'
TRAINING_MODE = True
SPLIT_RATIO = 0.85
RESAMPLE_INTERVAL = '1.5T'
N_STEPS_IN = 100
N_STEPS_OUT = 20

class IO:
	def data_import(resample = True, resample_interval = '2T'):
		'''Import JSON formatted data as dataframes and resample them to given intervals'''
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

	def data_export(df_future):
		'''Export prediction about future and mark the end of the programme'''
		df_future.to_json(Export_Data_Path + 'Predicted_Future_Values.json')

class Visualisation:
	def print_reference_data(df, y_test, y_test_predicted, n_steps_in, n_steps_out):
		'''Print out predicted and actual reference of the same time period for reference'''
		# Monitor dimensionality of output and test data
		print('\nThe dimensionality of test data is: ' + str(y_test.shape))
		print('The dimensionality of predicted data is: ' + str(predicted.shape) + '\n')

		# A sanity check on actual data
		np.set_printoptions(precision = 1, suppress = True)
		print('\n**********Actual ' + str(n_steps_out) + ' Tail Test Data**********')
		print(test_scaler.inverse_transform(y_test[-1]))
		print('**********End**********\n')
		print('**********Predicted ' + str(n_steps_out) + ' Tail Test Data**********')
		print(test_scaler.inverse_transform(predicted[-1]))
		print('**********End**********\n')
		print('**********Actual Tail Data For Deployment**********')
		print(df[- n_steps_in - n_steps_out:- n_steps_out])
		print('**********End**********\n')	
		
	def exploratory_plot_t1(df):
		'''Create correlation plots to explore necessity of modelling'''
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
		'''Provide a descriptive decomposition plot of raw data'''
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

	def rnn_history_plot(history):
		'''Provide a visual aid for efficiency of the RNN model'''
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
		'''Deprecated function'''
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
	'''Deprecated analytics class'''
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


	def deployment(z, model, train_scaler, test_scaler, n_steps_in, n_steps_out, columns, index, freq):
		'''Use existing model and the last known segment of data for prediction'''
		# Deploy the trained model to predict future
		future_predicted = test_scaler.inverse_transform(model.predict(z)[0])
		future_index = pd.date_range(start = index.array[-1], periods = n_steps_out + 1, freq = freq, closed = 'right')
		df_future = pd.DataFrame(future_predicted, columns = columns, index = future_index)

		# Print out future data for engineers' reference
		print('\n**********Predicted Future ' + str(n_steps_out) + ' Tail Data with ' + str(n_steps_in) + ' Lookback**********')
		print(df_future)
		print('**********End**********\n')
		
		# Return prediction about future for export
		return df_future

class Utilities:
	'''Auxillary functions'''
	def split_sequences(sequences, n_steps_in, n_steps_out):
		'''Lag data into difference sequences so to learn temporal structure within data'''
		X, y = list(), list()
		for i in range(len(sequences)):
			end_ix = i + n_steps_in
			out_end_ix = end_ix + n_steps_out
			if out_end_ix > len(sequences):
				break
			seq_x, seq_y = sequences[i:end_ix], sequences[end_ix: out_end_ix]
			X.append(seq_x)
			y.append(seq_y)
		# An unsupervised learning problem is now implemented as a supervised learning problem
		return np.array(X), np.array(y)

	def train_test_split(df_values, split_ratio):
		'''Split data into training and testing sets'''
		train_size = int(len(df_values) * split_ratio)
		test_size = len(df_values) - train_size
		train, test = df_values[0:train_size, :], df_values[train_size:len(df_values), :]
		print('\nTrain data size: ' + str(train_size) + '\n' + 'Test data size: ' + str(test_size) + '\n')
		return train, test

class Pre_Processing:
	def standardisation(df, n_steps_in, n_steps_out, split_ratio):
		'''Scale the data between 0 and 1 and split data into lagged input and output sequences'''
		values = df.values
		train_scaler = MinMaxScaler(feature_range = (0,1))
		test_scaler = MinMaxScaler(feature_range = (0,1))
		train, test = Utilities.train_test_split(values, split_ratio)
		train, test = train_scaler.fit_transform(train), test_scaler.fit_transform(test)
		z = np.array([test[-n_steps_in:]])
		X, y = Utilities.split_sequences(train, n_steps_in, n_steps_out)
		n_features = X.shape[2]
		# X is input data for training
		X = X.reshape((X.shape[0], X.shape[1], n_features))
		# y is output data for training
		y = y.reshape((y.shape[0], y.shape[1], n_features))
		# z is input data for later deployment
		z = z.reshape((z.shape[0], z.shape[1], n_features))
		return X, y, z, test, n_features, train_scaler, test_scaler

class Architecture:
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

class RNN:
	'''Main class responsible for execution of training and prediction'''
	def execution(df, training_mode, n_steps_in = 100, n_steps_out = 20, split_ratio = 0.8):
		'''Main function either to train or to import model ready for deployment'''
		# Scale data and split into train and test datasets
		X, y, z, test, n_features, train_scaler, test_scaler = Pre_Processing.standardisation(df, n_steps_in, n_steps_out, split_ratio)

		# Train or import model architecture, weights and configurations
		if training_mode:
			model, history = Architecture.architecture(X, y, n_steps_in, n_steps_out, n_features)
			Visualisation.rnn_history_plot(history)
		else:
			model = load_model('rnn_model.h5')

		# Visualize model structure to check for expected behaviour
		print(model.summary())
		plot_model(model, show_shapes = True, to_file = Export_Graph_Path + 'rnn_model.png')

		# Evaluate model accuracy
		x_test, y_test = Utilities.split_sequences(test, n_steps_in, n_steps_out)
		model.evaluate(x = x_test, y = y_test, callbacks = [EarlyStopping(monitor = 'loss')])
		y_test_predicted = model.predict(x_test)

		# Print out test and prediction data for reference
		Visualisation.print_reference_data(df, y_test, y_test_predicted, n_steps_in, n_steps_out)
		
		# Return a model, two scalers, two parameters and the input data z for model deployment
		return model, train_scaler, test_scaler, n_steps_in, n_steps_out, z

def main():
	# Import and resample JSON data into neural network required data format
	df, index, columns = IO.data_import(resample = True, resample_interval = RESAMPLE_INTERVAL)

	# Some helpful visualisation for engineers to guide engineers' intuition about data
	Visualisation.exploratory_plot_t1(df)
	Visualisation.all_factor_plot(df)

	# Standardise all data then either train or import model for deployment
	model, train_scaler, test_scaler, n_steps_in, n_steps_out, z = RNN.execution(df, training_mode = TRAINING_MODE, n_steps_in = N_STEPS_IN, n_steps_out = N_STEPS_OUT, split_ratio = SPLIT_RATIO)

	# Deploy the trained model obtained from last function to predict future
	df_future = RNN.deployment(z, model, train_scaler, test_scaler, n_steps_in, n_steps_out, columns, index, freq = RESAMPLE_INTERVAL)

	# Save the predicted future to JSON format
	IO.data_export(df_future)

if __name__ == '__main__':
	main()
