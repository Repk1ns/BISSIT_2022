import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2 as cv
import numpy as np
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

test_images = list()
test_images.append('test_images/0.jpeg')
test_images.append('test_images/1.jpeg')
test_images.append('test_images/2.jpeg')
test_images.append('test_images/3.jpeg')
test_images.append('test_images/4.jpeg')
test_images.append('test_images/5.jpeg')
test_images.append('test_images/6.jpeg')
test_images.append('test_images/7.jpeg')
test_images.append('test_images/8.jpeg')
test_images.append('test_images/9.jpeg')
test_images.append('test_images/0_t.jpeg')
test_images.append('test_images/9_d.png')
test_images.append('test_images/9_t.jpeg')

wrong = 0
good = 0


def find_digit(filename):
    for m in filename:
        if m.isdigit():
            return m

def evaluate(digit, filename):
    expected_number = find_digit(filename)
    expected_number = int(expected_number)
    predicted_number = digit[0]

    if expected_number == predicted_number:
        print(bcolors.OKGREEN + "CORRECT. Expected: {0}, Predicted: {1}".format(expected_number, predicted_number) + bcolors.ENDC)
        #good = good + 1
    else:
        print(bcolors.FAIL + "WRONG. Expected: {0}, Predicted: {1}".format(expected_number, predicted_number) + bcolors.ENDC)
        #wrong = wrong + 1
        

def test_all():
    for i in range(len(test_images)):

        img = load_img(test_images[i], color_mode="grayscale", target_size=(28, 28))

        img = img_to_array(img)

        img = img.reshape(1, 28, 28, 1)

        img = img.astype('float32')
        img = img / 255.0

        loaded_model = load_model("trained_model_2.h5")
        predict_value = loaded_model.predict(img)
        digit = predict_value.argmax(axis=-1)
        evaluate(digit, test_images[i])
        #print("Predicted number: {0}, Num in file: {1}".format(digit, test_images[i]))

def test_single():
    filename = 'test_images/0_t.jpeg'
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    plt.imshow(img)
    plt.show()

    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    loaded_model = load_model("trained_model_2.h5")
    predict_value = loaded_model.predict(img)
    digit = predict_value.argmax(axis=-1)
    #print("Predicted number: {0}".format(digit))
    evaluate(digit, filename)

if sys.argv[1] == 'all':
    test_all()
elif sys.argv[1] == 'single':
    test_single()
else:
    print("Wrong parameter")
    exit(42)

print("Right: {0}, Wrong: {1}".format(good, wrong))