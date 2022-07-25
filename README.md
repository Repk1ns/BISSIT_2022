# BISSIT_2022
Bissit 2022 - Team 1 - Handwritten Characters recognition

The projects consists of 2 source files. `main.py` and `train.py`.

- There are methods for load EMNIST dataset, preparing pixels in images (normalization etc), initialize neural network and training in the train.py.
- - Function `set_model()` prepare neural network. There are three convolution layers and densely-connected two layers.
- - Function `load_dataset()` loads mnist dataset.
- - Function `normalize()` converts integers to floats and normalize train and test data.
- - Function `train()` call all other functions and begin training.
- - Function `diagnostics()` shows an accuracy of training and evolution of the loss function in graph.

- In the main.py there are two functions for recognize handwritten character. 
- - First of them is `test_all()`. It tryies predict character for all images in `test_images` folder. All this images needs to be added in `test_images` list in `main.py`
- - Second function `test_single()` is for predicting one specific image - single calling `predict()`

Once training is done, trained model is stored in `trained_model.h5` file.

## How to run it?
```bash

# Training Neural Network
$ python train.py

# Predicting characters
# Runs predict for all images in test_images and test_images list
$ python main.py all

# Runs predict for single image (Need to change the path in test_single() function)
$ python main.py single
```

## Training graph
<img width="638" alt="Snímek obrazovky 2022-07-25 v 16 04 40" src="https://user-images.githubusercontent.com/43761153/180824390-da4530df-74a9-4af7-93b4-0201c5bf2600.png">

## Conclusion

Trained neural network could recognize a pretty, big and wide handwritten digits.
Loading and resizing the images will partially invalidate these images. The images are subsequently poorly recognized.

### Examples:

![0](https://user-images.githubusercontent.com/43761153/180827788-97c87710-eb39-4476-9069-f20fe48bddf8.jpeg)

🔴 NOT recognized - Predicted: 9 (Zero is probably too round)

![0_t](https://user-images.githubusercontent.com/43761153/180828117-1a4fd9f8-fcb4-44bc-85df-00a981cec54d.jpeg)

🟢 Recognized

![1](https://user-images.githubusercontent.com/43761153/180828174-d2d63bdc-581e-415c-bf30-af261d7e21e0.jpeg)

🟢 Recognized

![2](https://user-images.githubusercontent.com/43761153/180828209-fcfe35d1-9fa8-4ced-be10-7127bb1d601c.jpeg)

🟢 Recognized

![3](https://user-images.githubusercontent.com/43761153/180828230-c07d4f20-80d6-40c0-99ed-4c5db5d76857.jpeg)

🟢 Recognized

![4](https://user-images.githubusercontent.com/43761153/180828243-0e48f06d-ef98-4f90-a3c5-ec4371f17762.jpeg)

🟢 Recognized

![5](https://user-images.githubusercontent.com/43761153/180828381-8a1fe115-f9b9-4c9d-b3bb-a4121d1ebdc2.jpeg)

🟢 Recognized

![6](https://user-images.githubusercontent.com/43761153/180828401-a74bd84d-adf1-4ab5-ba7c-be820ab3fbc4.jpeg)

🟢 Recognized

![7](https://user-images.githubusercontent.com/43761153/180828454-fcbebc8d-76eb-4c8c-8e7e-006663d74c6e.jpeg)

🟢 Recognized

![8](https://user-images.githubusercontent.com/43761153/180828492-c7eb597c-ebdb-4722-adc4-535891ca3e0c.jpeg)

🟢 Recognized

<img width="195" alt="9_d" src="https://user-images.githubusercontent.com/43761153/180828529-a81deb2c-acb5-4c74-8b78-23cf0153755c.png">

🔴 NOT recognized - Predicted: 8 (Nine probably has too long tail)

![9](https://user-images.githubusercontent.com/43761153/180828586-9ba12611-38de-457f-bd7f-a987bea3ac22.jpeg)

🔴 NOT recognized - Predicted: 3 (Nine probably has too rounded tail)

![9_t](https://user-images.githubusercontent.com/43761153/180828610-5bb4e0c7-7954-4d5d-8f70-665fec6b53b1.jpeg)

🟢 Recognized
