import cv2
import os
import numpy as np
import random

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

Dataset = []
Target = []

# not waldos
for root, dirs, files in os.walk(r"C:\Users\Thomacdebabo\Documents\Git Folder PnS\Group-2-Waldo\Dataset\not waldo"):  
     for filename in files:
         print(filename)
          
         img = cv2.imread(r"C:\Users\Thomacdebabo\Documents\Git Folder PnS\Group-2-Waldo\Dataset\not waldo" + "\\" + filename, 1)
  
         img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         Dataset.append(img)
         Target.append(0.)

DatasetWaldo = []
TargetWaldo = []

# Waldos
for root, dirs, files in os.walk(r"C:\Users\Thomacdebabo\Documents\Git Folder PnS\Group-2-Waldo\Dataset\Waldo"):  
    for filename in files:
        print(filename)
        
        img = cv2.imread(r"C:\Users\Thomacdebabo\Documents\Git Folder PnS\Group-2-Waldo\Dataset\Waldo" + "\\" + filename, 1)

        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        DatasetWaldo.append(img)
        TargetWaldo.append(1.)

X = []
Y = []

X.extend(Dataset[:500])   
Y.extend(Target[:500])
X.extend(DatasetWaldo)
Y.extend(TargetWaldo)

X = np.array(X) / 255.
X = X - np.mean(X, axis=0)
Y = np.array(Y)

print(Y.shape)

DatasetWaldo = np.array(DatasetWaldo) / 255.
TargetWaldo = np.array(TargetWaldo)

Dataset = np.array(Dataset) / 255.
Target = np.array(Target)

print(Dataset.shape)
print(Target.shape)

# Define Model

x = Input(shape=(Dataset.shape[1], Dataset.shape[2], Dataset.shape[3]), name="input_layer")

conv1 = Conv2D(filters=30, kernel_size=(2, 2), strides=(2, 2), padding="same")(x) 
conv1 = Activation("relu")(conv1)

pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
pool2 = Activation("relu")(pool2)

conv2 = Conv2D(filters=25, kernel_size=(5, 5), strides=(2, 2), padding="same")(conv1)
conv2 = Activation("relu")(conv2)

pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

pool2 = Flatten()(pool2)

h1 = Dense(200)(pool2)
h1 = Activation("relu")(h1)

y = Dense(1, name="linear_layer")(h1)
y = Activation("sigmoid")(y)

model = Model(inputs=x, outputs=y)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])

# model.fit(
#     x=X, y=Y,
#     batch_size=12, epochs=10)
# 
# model.evaluate(Dataset, Target, 12, verbose=2)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    vertical_flip=True, width_shift_range=100, height_shift_range=100, rotation_range=30, zoom_range=0.2)
 
datagen.fit(X)
 
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X, Y, batch_size=12),
                    steps_per_epoch=100, epochs=1,
                    validation_data=datagen.flow(X, Y))

show = []
Targetshow = []

for i in range(5):
    rand = random.randint(0, Dataset.shape[0] - 1)
    show.append(Dataset[rand, :, :, :])
    Targetshow.append(Target[rand])
    
    rand = random.randint(0, DatasetWaldo.shape[0] - 1)
    show.append(DatasetWaldo[rand, :, :, :])
    Targetshow.append(TargetWaldo[rand])

show = np.array([show])

plt.figure()
for i in xrange(2):
    for j in xrange(5):
        y = model.predict(show[:, i * 5 + j, :, :, :])
        y = y[0]
        print y
        plt.subplot(2, 5, i * 5 + j + 1)
        plt.imshow(show[0, i * 5 + j])
        y0 = Targetshow[i * 5 + j]
        plt.title('Truth: {:.4} Prediction: {:.4}'.format(y0, y[0]))
plt.show()
