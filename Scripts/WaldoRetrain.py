#This script will retrain your existing Waldo model

import cv2
import os
import numpy as np
import random

from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Name of the existing model
ModelName = "WaldoS2"
#Directory of Dataset
DataDir = r"C:\Users\Thomacdebabo\Downloads\Hey-Waldo-master\Hey-Waldo-master\64"
#Size of Image Input
ImgSize = 64
#if there are too many Waldo pictures, we can set the amount of pictures with this
W = 100
#Ratio defines how many non Waldos per Waldos are used for training
ratio = 1

#some learning options
epochsteps=200
batch_size=20
maxepoch=20

#Dataset and DatasetWaldo are two lists where Waldos and non Waldos respectively are stored 
#(Dataset --> non Waldos / DatasetWaldo --> Waldos)

Dataset = []
Target = []

DatasetWaldo = []
TargetWaldo = []

# Waldos
for root, dirs, files in os.walk(DataDir+r"\Waldo"):  
    for filename in files:
        img = cv2.imread(DataDir+r"\Waldo" + "\\" + filename, 1)

        img = cv2.resize(img, (ImgSize, ImgSize), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        DatasetWaldo.append(img)
        TargetWaldo.append(1.)

#uncomment if you want to use all your Waldo pictures
#W = len(DatasetWaldo)
NW = int(W*ratio)

i = 1


# not waldos
for root, dirs, files in os.walk(DataDir+r"\not waldo"):  
    for filename in files:
        img = cv2.imread(DataDir + r"\not waldo" + "\\" + filename, 1)
  
        img = cv2.resize(img, (ImgSize, ImgSize), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Dataset.append(img)
        Target.append(0.)


X = []
Y = []
#chooses some Random samples from the whole dataset for training
for i in range(NW):
    rand = random.randint(0, len(Dataset) - 1)
    X.append(Dataset[rand])
    Y.append(Target[rand])
    
for i in range(W):
    rand = random.randint(0, len(DatasetWaldo) - 1)
    X.append(DatasetWaldo[rand])
    Y.append(TargetWaldo[rand])

X = np.array(X) / 255.
X = X - np.mean(X, axis=0)
Y = np.array(Y)
print(X.shape)

DatasetWaldo = np.array(DatasetWaldo) / 255.
TargetWaldo = np.array(TargetWaldo)

Dataset = np.array(Dataset) / 255.
Target = np.array(Target)

model = load_model(ModelName + ".hdf5")

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

waldo = optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay =0.0, amsgrad = False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    vertical_flip=True, width_shift_range=ImgSize/2, height_shift_range=ImgSize/2, rotation_range=20, fill_mode='wrap')
 
datagen.fit(X)
 
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
                    steps_per_epoch=epochsteps, epochs=maxepoch,
                    validation_data=datagen.flow(X, Y))

show = []
Targetshow = []

model.save(ModelName + ".hdf5")

for i in range(5):
    rand = random.randint(0, Dataset.shape[0] - 1)
    show.append(Dataset[rand, :, :, :])
    Targetshow.append(Target[rand])
    
    rand = random.randint(0, DatasetWaldo.shape[0] - 1)
    show.append(DatasetWaldo[rand, :, :, :])
    Targetshow.append(TargetWaldo[rand])

show = np.array([show])
Test = show - np.mean(show, axis=1)

plt.figure()
for i in range(2):
    for j in range(5):
        y = model.predict(Test[:, i * 5 + j, :, :, :])
        y = y[0]
        plt.subplot(2, 5, i * 5 + j + 1)
        plt.imshow(show[0, i * 5 + j])
        y0 = Targetshow[i * 5 + j]
        plt.title('Truth: {:.4} Prediction: {:.4}'.format(y0, y[0]))
plt.show()
