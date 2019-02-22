import keras as K
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import code
import time

from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Sequential, load_model

import matplotlib
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12
log_dir = './logs'

name = f"mnist-nn-128x2-{int(time.time())}"
# name = "mnist-nn-128x2-{}".format(int(time.time()))

mnist = tf.keras.datasets.mnist

(x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()

# we can skip this conversion if we use the
# sparse_categorical_crossentropy loss function instead of categorical_crossentropy
y_train = K.utils.to_categorical(y_train_raw, num_classes=10)
y_test = K.utils.to_categorical(y_test_raw, num_classes=10)


# normalize data
x_train = x_train/255
x_test = x_test/255

# code.InteractiveConsole(locals=globals()).interact()

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(10, activation = "softmax"))


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{name}", batch_size=batch_size)

model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size, callbacks=[tensorboard])

val_loss, val_acc = model.evaluate(x_test, y_test)

print("test data loss: " + str(val_loss))
print("test data accuracy: " + str(val_acc))

save_model = ""
while (save_model not in ["Y", "N"]):
    save_model = input("Would you like to save the model? (Y/N): ")
    save_model.strip()
    if (save_model == "Y"):
        model.save(name+".model")
        print(f"model {name} has been saved")


input("press enter to quit\n")
