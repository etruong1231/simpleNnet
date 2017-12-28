import numpy as np


class NeuralNet():


	def __init__(self,neurons,input_data,outputs_data):
		'''constructor for the neural network'''
		# inputs : layers neurons, input_data, output_Data

		# Creates a 1-input layer, 2-hiddenlayer, 1 ouput layer
		# input data to train the model that is feed through

		# output data to help evaluate the neural network

		np.random.seed(1)
		self.input = np.array(input_data)
		self.outputs = np.array(outputs_data)
		
		neuron_list = [self.input.shape[1]] + neurons
		# -1 to add a weight into each one
		self.weights = [2*np.random.random((neuron_list[x], neuron_list[x+1]))- 1 for x in range(3)]


	''' activation functions to help normalize the sums '''
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self,x):
		return x * (1-x)

	def feedForward(self, input):
		''' feed forward the data to train the neural net'''
		input_layer = self.input
		hidden_layer1 = self.sigmoid(np.dot(input_layer,self.weights[0]))
		hidden_layer2 = self.sigmoid(np.dot(hidden_layer1,self.weights[1]))
		output_layer = self.sigmoid(np.dot(hidden_layer2,self.weights[2]))
		return input_layer, hidden_layer1, hidden_layer2, output_layer


	def backPropagation(self,input_layer,hidden_layer1,hidden_layer2, output_layer,output_error,learning_rate):
		''' gets the each error on the layer and feed it backwards to readjust the neurons'''
		
		#checks how far off we are from the target
		output_delta = output_error * self.sigmoid_derivative(output_layer)
		hiddenL2_error = output_delta.dot(self.weights[2].T)
		hiddenL2_delta = hiddenL2_error * self.sigmoid_derivative(hidden_layer2)
		hiddenL1_error = hiddenL2_delta.dot(self.weights[1].T)
		hiddenL1_delta = hiddenL1_error * self.sigmoid_derivative(hidden_layer1)

		# need to adjust the weights to make it closer to the target 
		# learning rate moves a certain amount the descent.

		self.weights[2] += hidden_layer2.T.dot(output_delta) * learning_rate
		self.weights[1] += hidden_layer1.T.dot(hiddenL2_delta) * learning_rate
		self.weights[0] += input_layer.T.dot(hiddenL1_delta) * learning_rate
		

		      
	def predict(self,pred_input):
		''' after training it will be able to predict an outcome for the model'''
		input_layer = pred_input
		hidden_layer1 = self.sigmoid(np.dot(input_layer, self.weights[0]))
		hidden_layer2 = self.sigmoid(np.dot(hidden_layer1, self.weights[1]))
		output_layer = self.sigmoid(np.dot(hidden_layer2, self.weights[2]))
		return output_layer


	def train(self, learning_rate = 1, epochs = 60000, early_stopping = False):
		''' trains the neural network'''

		stopping = early_stopping
		previous_error = 100
		counter = 0

		# does a gradient descent
		for train_count in range(0,epochs):
			# feed the data forward the neural network
			input_layer, hidden_layer1, hidden_layer2, output_layer = self.feedForward(self.input)
			output_error = output_layer - self.outputs

			if (train_count % 10000) == 0:
				print("Epochs "+str(train_count)+"/"+str(epochs)+" Current Neural Network Error :"+str(np.mean(np.abs(output_error))))
				if(str(np.mean(np.abs(output_error))) < previous_error):
					previous_error = output_error
					counter = 0
				else:
					counter += 1
				if(stopping and previous_error < str(np.mean(np.abs(output_error))) and counter == 2):
					break

			# backPropagation will update the weights accordingly
			self.backPropagation(input_layer,hidden_layer1, hidden_layer2, output_layer,output_error,learning_rate)







if __name__ == "__main__":
	nNet = NeuralNet([4,3,1],[[0,1,0],[1,0,1],[0,1,0],[1,1,1]],[[0],[0],[0],[1]])
	nNet.train()
	print(nNet.predict([1,1,0]))















