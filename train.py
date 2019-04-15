"""
Program run command:

python train.py x_train.npy x_test.npy y_train.npy y_test.npy
"""
import sys
import numpy as np
import keras
from keras import layers, regularizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import layer_utils
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
from keras.models import load_model
from keras.initializers import glorot_uniform
from datetime import datetime

import keras.backend as K
K.set_image_data_format('channels_first')

#Channel size comes first in this dataset
def MiniModel(input_shape):

    X_input = Input(input_shape)
    # Layer 1: input = 3x112x112
    X = Conv2D(36,(8, 8), padding = 'valid', strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 1))(X_input)
    #X = Dropout(0.6)(X)
    X = BatchNormalization(axis = 1, name = 'bn1', momentum = 0.5)(X)
    X = Activation('relu')(X)

    # Layer 2: input = 36x53x53
    X = Conv2D(512,(8, 8), padding = 'valid', strides = (4, 4), name = 'conv2', kernel_initializer = glorot_uniform(seed = 2))(X)
    #X = Dropout(0.6)(X)
    X = BatchNormalization(axis = 1, name = 'bn2', momentum = 0.5)(X)
    X = Activation('relu')(X)

    # Layer 3: input = 512x12x12
    X = MaxPooling2D(pool_size = (8, 8), strides = (4, 4), name = 'max_pool1')(X)

    # Layer 2: input = 512x2x2
    # Flatten -> FC
    X = Flatten()(X)
    X = Dense(10, activation = 'softmax', name = 'fc1', kernel_initializer = glorot_uniform(seed = 3), kernel_regularizer = regularizers.l2(0.025))(X)

    model = Model(inputs = X_input, outputs = X, name = 'Model')

    return model

folder = 'MiniImageNet'
# Load files
X_train_original = np.load(sys.argv[1])
#X_test_original = np.load(sys.argv[2])
Y_train_original = np.load(sys.argv[2])
#Y_test_original = np.load(sys.argv[4])

# Normalize image vectors
X_train = X_train_original/255.
#X_test = X_test_original/255.

# One hot encoding
Y_train = to_categorical(Y_train_original, 10)
#Y_test = to_categorical(Y_test_original, 10)

# Print Shapes
print ("number of training examples = " + str(X_train.shape[0]))
#print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
#print ("X_test shape: " + str(X_test.shape))
#print ("Y_test shape: " + str(Y_test.shape))


# Initialize and compile model
miniModel = MiniModel(X_train.shape[1:])
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-2)

miniModel.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Start optimization
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

csv_logger = CSVLogger(folder + time + '_cost_hw6.csv' , append=True, separator=',')
miniModel.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 64, callbacks=[csv_logger])
#miniModel.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 64, validation_data = (X_test, Y_test), callbacks=[csv_logger])


print("Save model?(y/n)")
input1 = input()
if (input1 =='y'):
    miniModel.save(sys.argv[3])
"""
preds = miniModel.evaluate(x = X_test, y = Y_test, validation_data = (X_test, Y_test), sample_weight = None)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
"""

"""
print("Save model?(y/n)")
input1 = input()
if (input1 =='y'):
    miniModel.save(folder + time + '_miniModel.hd5')
"""
