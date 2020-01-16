import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import Constant
from keras import backend as K
from keras.layers import ELU
import matplotlib.pyplot as plt
import numpy as np

# Model configuration
img_width, img_height = 32, 32
batch_size = 250
no_epochs = 100
no_classes = 100
validation_split = 0.2
verbosity = 1
elu_alpha = 0.1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = cifar100.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0], 1, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 3)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 3)
    input_shape = (img_width, img_height, 3)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Convert them into black or white: [0, 1].
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(no_classes, activation='softmax', kernel_initializer='he_normal'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history = model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)
model_relu = model

# Define
t = -1.0
def ftswish(x):
  return K.maximum(t, K.relu(x)*K.sigmoid(x) + t)

# Create the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation=ftswish, input_shape=input_shape, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), activation=ftswish, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, kernel_initializer='he_normal', activation=ftswish))
model.add(Dense(256, kernel_initializer='he_normal', activation=ftswish))
model.add(Dense(no_classes, activation='softmax', kernel_initializer='he_normal'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history_FTSwish = model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)
model_ftswish = model

# Define
t = -1.0
def sigmoid(x):
  return (1 / (1 + (np.e**-x)))
def swish(x):
  return x * K.sigmoid(x)

# Create the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, activation=swish, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3, 3), activation=swish, kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation=swish, kernel_initializer='he_normal'))
model.add(Dense(256, activation=swish, kernel_initializer='he_normal'))
model.add(Dense(no_classes, activation='softmax', kernel_initializer='he_normal'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history_Swish = model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)


# Generate generalization metrics
score = model_relu.evaluate(input_test, target_test, verbose=0)
print(f'Test loss for Keras ReLU CNN: {score[0]} / Test accuracy: {score[1]}')
score = model_ftswish.evaluate(input_test, target_test, verbose=0)
print(f'Test loss for Keras FTSwish CNN: {score[0]} / Test accuracy: {score[1]}')
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss for Keras Swish CNN: {score[0]} / Test accuracy: {score[1]}')

# Visualize model history
plt.plot(history_FTSwish.history['accuracy'], label='Training accuracy')
plt.plot(history_FTSwish.history['val_accuracy'], label='Validation accuracy')
plt.title('FTSwish training / validation accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

plt.plot(history_FTSwish.history['loss'], label='Training loss')
plt.plot(history_FTSwish.history['val_loss'], label='Validation loss')
plt.title('FTSwish training / validation loss values')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

# Compare validation loss
plt.plot(history.history['val_loss'], label='ReLU training loss')
plt.plot(history_Swish.history['val_loss'], label='Swish validation loss')
plt.plot(history_FTSwish.history['val_loss'], label='FTSwish validation loss')
plt.title('ReLU, Swish & FTSwish validation losses')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()