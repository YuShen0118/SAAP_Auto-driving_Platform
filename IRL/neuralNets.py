"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
from keras import backend as K
import keras


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def net1(numInputs, numOutputs, params, weightsFile='', epochCount=1, enlarge_lr=0):
    netInputs = Input(shape=(numInputs,))
    x = Dense(params[0], kernel_initializer='lecun_uniform', activation='relu')(netInputs)
    x = Dropout(0.2)(x)
    x = Dense(params[1], kernel_initializer='lecun_uniform', activation='relu')(x)
    x = Dropout(0.2)(x)
    netOutputs = Dense(numOutputs, kernel_initializer='lecun_uniform', activation='linear')(x)

    model = Model(inputs = netInputs, outputs = netOutputs)
    
    #lr = 0.001 / 2**(epochCount-1)
    #print('===============lr===============', lr)

    #optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    #optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #optimizer = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    #optimizer = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mse')
    
    if weightsFile:
        model.load_weights(weightsFile)

    return model


def net1_old(numFeatures, numNodes, weights=''):
    model = Sequential()

    # First layer.
    model.add(Dense(numNodes[0], input_shape=(numFeatures,), kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.2))

    # Second layer.
    model.add(Dense(numNodes[1], kernel_initializer='lecun_uniform', activation='relu'))
    model.add(Dropout(0.2))

    # Output layer.
    model.add(Dense(3, kernel_initializer='lecun_uniform', activation='linear'))

    model.compile(optimizer=RMSprop(), loss='mse')

    if weights:
        model.load_weights(weights)

    return model


def lstm_net_old(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(
        output_dim=512, input_dim=num_sensors, return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=512, input_dim=512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=3, input_dim=512)) #!
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model
