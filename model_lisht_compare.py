import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras_contrib.callbacks.dead_relu_detector  import DeadReluDetector

# LiSHT
def LiSHT(x):
  return x * K.tanh(x)

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 2
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0], 1, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data: [0, 1].
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation=LiSHT, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation=LiSHT))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation=LiSHT))
model.add(Dense(no_classes, activation='softmax'))

model_com = Sequential()
model_com.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model_com.add(MaxPooling2D(pool_size=(2, 2)))
model_com.add(Dropout(0.25))
model_com.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_com.add(MaxPooling2D(pool_size=(2, 2)))
model_com.add(Dropout(0.25))
model_com.add(Flatten())
model_com.add(Dense(256, activation='relu'))
model_com.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model_com.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Define callbacks
callbacks = [
  DeadReluDetector(x_train=input_train, verbose=True)
]

# Fit data to model
history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)
history_com = model_com.fit(input_train, target_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_split=validation_split,
                callbacks=callbacks)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'LiSHT - Test loss: {score[0]} / Test accuracy: {score[1]}')
score_com = model_com.evaluate(input_test, target_test, verbose=0)
print(f'ReLU - Test loss: {score_com[0]} / Test accuracy: {score_com[1]}')

# Plot history: Crossentropy loss
plt.plot(history.history['val_loss'], label='LiSHT')
plt.plot(history_com.history['val_loss'], label='ReLU')
plt.title('Crossentropy validation loss for LiSHT and ReLU')
plt.ylabel('Loss value')
plt.xlabel('Epochs')
plt.legend(loc="upper left")
plt.show()

# Plot history: Accuracies
plt.plot(history.history['val_accuracy'], label='LiSHT')
plt.plot(history_com.history['val_accuracy'], label='ReLU')
plt.title('Accuracies for LiSHT and ReLU')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc="upper left")
plt.show()