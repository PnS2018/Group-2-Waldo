#This is a Benchmark to Test a Model on a specific Picture

import cv2
import numpy as np

from keras.models import load_model

DataDir = r"C:\Users\Thomacdebabo\Downloads\Dataset(1)\Dataset\Controll\2.jpg"
ModelName = "model2_10200epochs.h5"

ImgSize = 224

img = cv2.imread(DataDir, 1)
a = img.shape
n = 10
size = a[1] / n
m = a[0] / size
m = int(m)
size = int(size)
hsize = int(size/2)

img = cv2.resize(img, (size * n, size * m))

pics = []
picso= []
#We subdivide the Picture twice, one time we shift the picture by half the cropped image size
for i in range(n):
    for j in range(m):
        cr = img[j * size:(j + 1) * size, i * size:(i + 1) * size]
        cr = cv2.resize(cr, (ImgSize, ImgSize), interpolation=cv2.INTER_CUBIC)
        cr = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)
        pics.append([cr])
        
for i in range(n-1):
    for j in range(m-1):
        cr = img[(j * size+hsize):((j + 1) * size+hsize),( i * size+hsize):((i + 1) * size+hsize)]
        cr = cv2.resize(cr, (ImgSize, ImgSize), interpolation=cv2.INTER_CUBIC)
        cr = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)
        picso.append([cr])
        

pics = np.array(pics) / 255.
pics = pics - np.mean(pics, axis=0)

picso = np.array(picso) / 255.
picso = pics - np.mean(picso, axis=0)

print(pics.shape)
print(n)
print(m)

model = load_model(ModelName)

print("[MESSAGE] Model is defined.")

#The Prediction of each section is multiplied with the RGB values of the section
for i in range(n):
    for j in range(m):
        img[j * size:(j + 1) * size, i * size:(i + 1) * size] = img[j * size:(j + 1) * size, i * size:(i + 1) * size] * model.predict(pics[i * m + j])

for i in range(n-1):
    for j in range(m-1):
        img[(j * size+hsize):((j + 1) * size+hsize),( i * size+hsize):((i + 1) * size+hsize)] = img[(j * size+hsize):((j + 1) * size+hsize),( i * size+hsize):((i + 1) * size+hsize)] * model.predict(picso[i * m + j])


img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

cv2.imshow("Wo isch de Walti?", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
