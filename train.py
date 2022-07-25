import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

###########
# LOAD THE EMNIST DATASET
# RESHAPE DATASET TO HAVE ONLY ONE CHANNEL
# ONE HOT ENCODE
###########
def load_dataset():
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

	Y_train = to_categorical(Y_train)
	Y_test = to_categorical(Y_test)

	return X_train, Y_train, X_test, Y_test

###########
# CONVERT INTEGERS TO FLOATS
# NORMALIZE
###########
def normalize(train, test):
	train_f = train.astype('float32')
	test_f = test.astype('float32')

	normalized_train = train_f / 255.0
	normalized_test = test_f / 255.0

	return normalized_train, normalized_test

###########
# DEFINE NEURAL NETWORK MODEL
###########
def set_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=(1,1), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # ANOTHER OPTIMIZER (ADAM)
    # model.compile(loss="categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["acc"])

    model.summary()
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    return model


###########
# SUMMARIZE DIAGNOSTICS
###########
def diagnostics(history):

	plt.subplot(2, 1, 1)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	plt.subplot(2, 1, 2)
	plt.title('Classification Accuracy')
	plt.plot(history.history['acc'], color='blue', label='train')
	plt.plot(history.history['val_acc'], color='orange', label='test')
	plt.show()
 

###########
# SUMMARIZE PERFORMANCE
###########
def performance(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	plt.boxplot(scores)
	plt.show()


###########
# LOAD DATASED
# PREPARE IMAGE - TYPE CONVERSE, NORMALIZE
# DEFINE NEURAL NETWORK MODEL
# TRAIN NEURAL NETWORK
# SAVE MODEL
###########
def train():
    X_train, Y_train, X_test, Y_test = load_dataset()
    X_train, X_test = normalize(X_train, X_test)
    model = set_model()
    history = model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)
    loss, acc = model.evaluate(X_test, Y_test)
    diagnostics(history)
    print("Accuracy: {0}, Loss: {1}".format(acc, loss))
    model.save('trained_model_2.h5')
    

train()

print("Model saved into trained_model.h5")
