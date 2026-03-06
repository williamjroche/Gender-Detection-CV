import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib.pyplot as plt

#specifications
img_height = 128
img_width = 128
img_channels = 3
batch_size = 32
epochs = 20
male_count = 0
female_count = 0

#training data - change file path to your training set
train_dataset = keras.utils.image_dataset_from_directory(
    'some file path',
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
print(train_dataset.class_names)

#validation data - change file path to your validation set
eval_dataset = keras.utils.image_dataset_from_directory(
    'some file path',
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
print(eval_dataset.class_names)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.2),
    keras.layers.RandomContrast(0.2)
])

#define model, this one is a Convolution Neural Network with a basic architecture
model = keras.Sequential([
    keras.layers.Input(shape=[img_height, img_width, img_channels]),
    data_augmentation,
    keras.layers.Rescaling(1./255),

    keras.layers.Conv2D(16, (3,3), activation='relu'),
    keras.layers.BatchNormalization(), #so we don't get overfitting
    keras.layers.MaxPooling2D(pool_size=2, strides=2),

    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=2, strides=2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'), #gradually decrease nodes to avoid information loss
    keras.layers.Dropout(0.5), #dropout after the layer to avoid weights getting too big for any one node
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(2, activation='softmax')
])

model.summary() #prints model architecture (layer type, shape, parameter #, total trainable parameters)

model.compile( #optimizer and loss
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train for more epochs with early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

model.fit( #specify train dataset, validation, and epochs
    train_dataset,
    validation_data=eval_dataset,
    epochs=epochs,
    callbacks=[early_stopping]
)


model.save('some file path') #save the model to your needed path