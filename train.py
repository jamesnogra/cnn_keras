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

TRAIN_DIR = 'train'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 25
MODEL_NAME = 'keras-fruits'

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

def create_train_data(all_classes, all_labels):
	training_data = []
	for label_index, specific_class in enumerate(all_classes):
		current_dir = TRAIN_DIR + '/' + specific_class
		print('Reading directory of ' + current_dir)
		for img_filename in os.listdir(current_dir):
			path = os.path.join(current_dir, img_filename)
			img = cv2.imread(path)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			training_data.append([np.array(img), [all_labels[label_index]]])
	shuffle(training_data)
	return training_data

all_classes, NUM_OUTPUT = define_classes()
all_labels = define_labels(all_classes)
training_data = create_train_data(all_classes, all_labels)

# Define training and testing data
train = training_data[:int(len(training_data)*(PERCENT_TRAINING_DATA/100))]
test = training_data[-int(len(training_data)*(PERCENT_TRAINING_DATA/100)):]
train_images = np.array([i[0] for i in train])
train_labels = np.array([i[1] for i in train])
test_images = np.array([i[0] for i in test])
test_labels = np.array([i[1] for i in test])
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

'''# Plot 25 images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(all_classes[train_labels[i][0]])
#.show()'''

# Make the model
model = models.Sequential()
model.add(layers.Conv2D(FIRST_NUM_CHANNEL, (FILTER_SIZE, FILTER_SIZE), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.MaxPooling2D((FILTER_SIZE, FILTER_SIZE)))
model.add(layers.Conv2D(FIRST_NUM_CHANNEL*2, (FILTER_SIZE, FILTER_SIZE), activation='relu'))
model.add(layers.MaxPooling2D((FILTER_SIZE, FILTER_SIZE)))
model.add(layers.Conv2D(FIRST_NUM_CHANNEL*4, (FILTER_SIZE, FILTER_SIZE), activation='relu'))
model.add(layers.MaxPooling2D((FILTER_SIZE, FILTER_SIZE)))
model.add(layers.Conv2D(FIRST_NUM_CHANNEL*8, (FILTER_SIZE, FILTER_SIZE), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(FIRST_NUM_CHANNEL*16, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(NUM_OUTPUT))
model.summary()

# Train the model
model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)
history = model.fit(
	train_images,
	train_labels,
	epochs=NUM_EPOCHS, 
	validation_data=(test_images, test_labels)
)

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
plt.show()

# Save model
model.save(MODEL_NAME)

# Write classes to js inside web-codes
# This will be used for tensorflowjs at index.html
f = open('web-codes/classes.js', 'w')
all_classes_str = ''
for class_label in all_classes:
	all_classes_str += '"' + class_label + '",'
f.write('const allClasses = [' + all_classes_str + ']')
f.close()