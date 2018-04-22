import cv2
import os
import numpy as np
import random

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

DataDir = r"F:\Waldo\Hey-Waldo-master\64"
ModelName = "WaldoSmall2"

ratio = 1.2
Dataset = []
Target = []

# not waldos
for root, dirs, files in os.walk(DataDir + r"\not waldo"):  
    for filename in files:
        print(filename)
          
        img = cv2.imread(DataDir + r"\not waldo" + "\\" + filename, 1)
  
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Dataset.append(img)
        Target.append(0.)

DatasetWaldo = []
TargetWaldo = []

# Waldos
for root, dirs, files in os.walk(DataDir + r"\Waldo"):  
    for filename in files:
        print(filename)
        
        img = cv2.imread(DataDir + r"\Waldo" + "\\" + filename, 1)

        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        DatasetWaldo.append(img)
        TargetWaldo.append(1.)


W = len(DatasetWaldo)
NW = int(W*ratio)
X = []
Y = []
for i in range(NW):
    rand = random.randint(0, len(Dataset) - 1)
    X.append(Dataset[rand])
    Y.append(Target[rand])
    
X.extend(DatasetWaldo)
Y.extend(TargetWaldo)

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

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])

# model.fit(
#     x=X, y=Y,
#     batch_size=12, epochs=10)
# 
# model.evaluate(Dataset, Target, 12, verbose=2)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    vertical_flip=True, width_shift_range=200, height_shift_range=200, rotation_range=20, zoom_range=0.2, fill_mode='wrap')
 
datagen.fit(X)
 
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X, Y, batch_size=10),
                    steps_per_epoch=20, epochs=30,
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
