import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

# Paths
train_dir = "data/train"
val_dir = "data/val"
IMG_SIZE = (250, 250)
BATCH_SIZE = 32
EPOCHS = 20 
NUM_CLASSES = len(os.listdir(train_dir))

# Data generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    shear_range=0.3,
    brightness_range=(0.7, 1.3),
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

# Model base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(250, 250, 3))

# Fine-tuning: freeze all except last 40 layers
for layer in base_model.layers[:-40]:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

# Classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


# Callbacks
checkpoint = ModelCheckpoint('models/rice_model.h5', save_best_only=True)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Training
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
   
)
