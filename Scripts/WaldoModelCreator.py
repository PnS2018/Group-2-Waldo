#This script will create a Machine Learning Model to dedect Waldo
#You have to provide the Directory where the Dataset is located and a Name for the Model
#there are some Variables which are tweakable very easily
import cv2
import os
import numpy as np
import random
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Directory of Dataset. Has to have a folder "waldo" and "not waldo" with the pictures
DataDir = r"C:\Users\Thomacdebabo\Downloads\Hey-Waldo-master\Hey-Waldo-master\64"
#Model will be saved under [ModelName].hdf5
ModelName = "WaldoS2"
#determine how large your input pictures should be (if pictures in the Dataset don't have this size, they will be resized)
ImgSize = 64
#Ratio defines how many non Waldos per Waldos are used for training
ratio = 2

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

W = len(DatasetWaldo)
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

#X and Y are our final Trainingset
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


# Define Model

x = Input(shape=(Dataset.shape[1], Dataset.shape[2], Dataset.shape[3]), name="input_layer")

conv1 = Conv2D(filters=50, kernel_size=(3, 3), strides=(3, 3), padding="same")(x) 
conv1 = Activation("relu")(conv1)

conv2 = Conv2D(filters=25, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv1)
conv2 = Activation("relu")(conv2)

conv3 = Conv2D(filters=20, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv2)
conv3 = Activation("relu")(conv3)

dropout = Dropout(rate=0.2)(conv3)
dropout = Flatten()(dropout)

h1 = Dense(75)(dropout)
h1 = Activation("relu")(h1)
dropout2 = Dropout(rate=0.5)(h1)

y = Dense(1, name="linear_layer")(dropout2)
y = Activation("sigmoid")(y)

model = Model(inputs=x, outputs=y)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()
waldo = optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay =0.0, amsgrad = False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


datagen = ImageDataGenerator( 
    featurewise_center=True,
    featurewise_std_normalization=False,
    vertical_flip=True, width_shift_range=200, height_shift_range=200, rotation_range=20, fill_mode= 'wrap')
 
datagen.fit(X)
 
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X, Y, batch_size=5),
                    steps_per_epoch=100, epochs=10, shuffle = True,
                    validation_data=datagen.flow(X, Y), class_weight={0: 1, 1: ratio})



#Saving the Model
model.save(ModelName +".hdf5")

#Showing some examples to personally evaluate the model
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
Test = show - np.mean(show, axis=1)

plt.figure()
for i in range(2):
    for j in range(5):
        y = model.predict(Test[:, i * 5 + j, :, :, :] )
        y = y[0]
        plt.subplot(2, 5, i * 5 + j + 1)
        plt.imshow(show[0, i * 5 + j])
        y0 = Targetshow[i * 5 + j]
        plt.title('Truth: {:.4} Prediction: {:.4}'.format(y0, y[0]))
plt.show()
