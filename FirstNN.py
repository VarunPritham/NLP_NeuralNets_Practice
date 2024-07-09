import tensorflow as tf
import os

import numpy as np

os.makedirs('data', exist_ok=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=os.path.join(os.getcwd(), 'data', 'mnist.npz'))
#
print(f"(Before) x_train has shape: {x_train.shape}")
print(f"(Before) x_test has shape: {x_test.shape}")

print(f"(Before) x_train has - min: {np.min(x_train)}, max: {np.max(x_train)}")
print(f"(Before) x_test has - min: {np.min(x_test)}, max: {np.max(x_test)}")

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(f"(After) x_train has shape: {x_train.shape}")
print(f"(After) x_test has shape: {x_test.shape}")

x_train = (x_train - np.mean(x_train, axis=1, keepdims=True))/np.std(x_train, axis=1, keepdims=True)
x_test = (x_test - np.mean(x_test, axis=1, keepdims=True))/np.std(x_test, axis=1, keepdims=True)

print(f"(After) x_train has - min: {np.min(x_train)}, max: {np.max(x_train)}")
print(f"(After) x_test has - min: {np.min(x_test)}, max: {np.max(x_test)}")

batch_size = 100
img_width, img_height = 28,28
input_size = img_height * img_width
num_labels = 10

def mnist_model():
    """ Defining the model using Keras sequential API """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Defining the model, optimizer and a loss function
model = mnist_model()
optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])

NUM_EPOCHS = 10

""" Model training """

# Creating onehot encoded labels
y_onehot_train = np.zeros((y_train.shape[0], num_labels), dtype=np.float32)
y_onehot_train[np.arange(y_train.shape[0]), y_train] = 1.0

# Training Phase
train_history = model.fit(x_train,y_onehot_train, batch_size=batch_size, epochs=NUM_EPOCHS, validation_split=0.2)

""" Testing phase """

# Test inputs and targets
y_onehot_test = np.zeros((y_test.shape[0], num_labels), dtype=np.float32)
y_onehot_test[np.arange(y_test.shape[0]), y_test] = 1.0

# Evaulte on test data
test_res = model.evaluate(x_test, y_onehot_test, batch_size=batch_size)

print("Testing Results: ")
print(f"\tLoss: {test_res[0]}")
print(f"\tAccuracy: {test_res[1]}")