import os
import numpy as np
# pip install opencv-python
import cv2
from random import shuffle
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
# Keras
from tensorflow.keras import datasets, layers, models
from tensorflow import keras

TEST_IMG = 'default.jpg'

TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 15
MODEL_NAME = 'keras-main-model'

def define_classes():
	all_classes = []
	for folder in os.listdir(TRAIN_DIR):
		all_classes.append(folder)
	return all_classes, len(all_classes)

def define_labels(all_classes):
	all_labels = []
	for x in range(len(all_classes)):
		all_labels.append(x)
	return all_labels

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)

# Load saved model
model = keras.models.load_model(MODEL_NAME)
#model.summary()

# Test the model
img = tf.keras.utils.load_img(
	TEST_IMG, target_size=(IMG_SIZE, IMG_SIZE)
)
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = np.array(tf.nn.softmax(predictions[0]))
for x in range(len(all_labels)):
	print(all_classes[x] + ' ' + '{:.4f}'.format(score[x]))