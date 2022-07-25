import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

input_shape = X_train.shape[1] * X_train.shape[2]
number_of_classes = len(set(y_train))

X_train = X_train / 255.0
X_test = X_test / 255
X_train = X_train.reshape(-1, input_shape)
X_test = X_test.reshape(-1, input_shape)

y_train = to_categorical(y_train, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

model = Sequential()
model.add(Dense(128, activation="relu", input_shape=X_train.shape[1:]))
model.add(Dense(y_train.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
loss, acc = model.evaluate(X_test, y_test)
print("LOSS:", loss)
print("ACCURACY:", acc)

model.save("model.h5")
loaded_model = load_model("model.h5")

y_prob = model.predict(X_test[:1])[0]
pred = y_prob.argmax(axis=-1)

print("real:", y_test[0].argmax())
print("predict:", pred) 