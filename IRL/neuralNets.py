"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
from keras.optimizers import Adam


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def net1(numInputs, numOutputs, params, weightsFile='', epochCount=1, enlarge_lr=0):
    netInputs = Input(shape=(numInputs,))
    #x = BatchNormalization()(netInputs)
    x = Dense(params[0], kernel_initializer='lecun_uniform', activation='relu')(netInputs)
    x = Dropout(0.2)(x)

    for i in range(1, len(params)):
        #x = BatchNormalization()(x)
        x = Dense(params[i], kernel_initializer='lecun_uniform', activation='relu')(x)
        x = Dropout(0.2)(x)
    netOutputs = Dense(numOutputs, kernel_initializer='lecun_uniform', activation='linear')(x)

    model = Model(inputs = netInputs, outputs = netOutputs)
    
    model.compile(optimizer='rmsprop', loss='mse')

    #model.summary()

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
