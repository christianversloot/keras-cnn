import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.constraints import max_norm

# Model configuration
img_width, img_height         = 32, 32
batch_size                    = 250
no_epochs                     = 55
no_classes                    = 10
validation_split              = 0.2
verbosity                     = 1
max_norm_value                = 2.0

# Load CIFAR10 dataset
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0],3, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 3, img_width, img_height)
    input_shape = (3, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 3)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 3)
    input_shape = (img_width  , img_height, 3)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split
)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# ============
# Test loss: 0.8185625041961669 / Test accuracy: 0.7193999886512756
# ============
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ============
# Test loss: 1.3634590747833253 / Test accuracy: 0.5906000137329102
# ============
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(Dropout(0.50))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.50))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.50))
# model.add(Dense(no_classes, activation='softmax'))

# ============
# Test loss: 0.8021318348884583 / Test accuracy: 0.7243000268936157
# ============
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ============
# Test loss: 1.0652880083084106 / Test accuracy: 0.623199999332428
# ============
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ============
# Test loss: 0.7692169213294983 / Test accuracy: 0.7314000129699707
# ============
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(2.0), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(2.0), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(2.0), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# Test loss: 0.8035776728630066 / Test accuracy: 0.7233999967575073
# maxnorm=1.0
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# Test loss: 0.8003060577392578 / Test accuracy: 0.7250000238418579
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# Test loss: 0.7669966766357422 / Test accuracy: 0.7365999817848206
# maxnorm = 2.5
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

#Test loss: 2.3026647621154783 / Test accuracy: 0.10000000149011612
# maxnorm = 2.5
# lr 10e-2 && decay linear
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

#Test loss: 2.302865937805176 / Test accuracy: 0.10000000149011612
#maxnorm = 2.5
# lre0 && decay linear
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ==================
# Test loss: Test loss: 0.9980722076416015 / Test accuracy: 0.6534000039100647
# SGD, momentum 0.99, Nesterov false, LR 10e-2, LR decay linear
# maxnorm=2.5
# ==================
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ==================
# Test loss: Test loss: 0.965770835018158 / Test accuracy: 0.6678000092506409
# SGD, momentum 0.99, Nesterov true, LR 10e-2, LR decay linear
# maxnorm=2.5
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ==================
# Test loss: Test loss: 1.0010871562957764 / Test accuracy: 0.6502000093460083
# SGD, momentum 0.99, Nesterov true, default LR settings  
# MAXNORM=2.5
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))

# ==================
# Test loss: Test loss: 0.9282757438659668 / Test accuracy: 0.6773999929428101
# SGD, momentum 0.99, Nesterov true, default LR settings  
# MAXNORM=2.0
# model = Sequential()
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_constraint=max_norm(max_norm_value), kernel_initializer='he_uniform'))
# model.add(Dense(no_classes, activation='softmax'))