import keras as K
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np
import code
import time
import os
import glob
from termcolor import colored, cprint

from keras.layers import Input, Flatten, Dense, Activation
from keras.models import Sequential, load_model

import matplotlib
matplotlib.use('MacOSX')

import matplotlib.pyplot as plt

files = glob.glob("./*.model")
files.sort(key=os.path.getmtime)

available_models = []

for file in reversed(files):
    filename = os.path.basename(file)
    name, ext = os.path.splitext(filename)
    available_models.append(name)

cprint("Available Models:", "yellow")
for idx, file in enumerate(available_models):
    print(colored(f"{idx+1}. {file}", 'cyan'))

valid_range = range(1,len(files)+1)

def prompt_selection():
    return input(colored("\nwhich model would you like to load? ", 'green'))

result = prompt_selection()

while (not result.isdigit() or int(result) not in valid_range):
    cprint(f"\ninvalid selection, please select a number between 1 and {len(files)+1}", 'red')
    result = prompt_selection()
    result.strip()

print("you selected " + str(result))

model_name = available_models[int(result) - 1] + ".model"
model = load_model(model_name)

mnist = K.datasets.mnist

(x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()
y_train = K.utils.to_categorical(y_train_raw, num_classes=10)
y_test = K.utils.to_categorical(y_test_raw, num_classes=10)

# normalize data
x_train = x_train/255
x_test = x_test/255

predictions = model.predict([x_test])

loss, acc = model.evaluate(x_test, y_test)

print("test data loss: " + str(loss))
print("test data accuracy: " + str(acc))
