"""
Program run command:

python test.py miniModel.hd5 x_test.npy y_test.npy
"""
import sys
import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model

import keras.backend as K
K.set_image_data_format('channels_first')


# Load files
X_test_original = np.load(sys.argv[1])
Y_test_original = np.load(sys.argv[2])

# Normalize image vectors
X_test = X_test_original/255.

# One hot encoding
Y_test = to_categorical(Y_test_original, 10)

# Print Shapes
print("number of test examples = " + str(X_test.shape[0]))

# Load model
miniModel = load_model(sys.argv[3])

# Model summary
print(miniModel.summary())

# Evaluate test set
preds = miniModel.evaluate(x = X_test, y = Y_test, sample_weight = None)

print("Test Error = " + str((1-preds[1]) * 100), '%')
print("Test Accuracy = " + str(preds[1] * 100), '%')
