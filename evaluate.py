import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

class Evaluator():
    def recognize(X_test, y_test, model):
        loaded_model = load_model("model.h5")
    
        plt.imshow(X_test[0].reshape(28, 28))
        plt.show()
    
        y_prob = model.predict(X_test[:1])[0]
        pred = y_prob.argmax(axis=-1)
    
        print("real:", y_test[0].argmax())
        print("predict:", pred) 