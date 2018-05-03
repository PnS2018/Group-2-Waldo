import cv2
import numpy as np

from keras.models import load_model

DataDir = r"F:\Waldo\Dataset\Controll\2.jpg"
ModelName = "WaldoSmall2"

img = cv2.imread(DataDir,1)
a = img.shape
n= 100
size = a[1]/n
m = a[0]/size
m = int(m)
size = int(size)

img = cv2.resize(img, (size*n,size*m))

pics = []
for i in range(n):
    for j in range(m):
        cr = img[j*size:(j+1)*size, i*size:(i+1)*size]
        cr = cv2.resize(cr,(200,200), interpolation=cv2.INTER_CUBIC)
        cr = cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)
        pics.append([cr])
        

pics = np.array(pics)/255.
pics = pics -np.mean(pics, axis=0)

print(pics.shape)
print(n)
print(m)

model = load_model(ModelName + ".hdf5")

print("[MESSAGE] Model is defined.")

for i in range(n):
    for j in range(m):
        img[j*size:(j+1)*size, i*size:(i+1)*size]= img[j*size:(j+1)*size, i*size:(i+1)*size]*model.predict(pics[i*m+j])

img = cv2.resize(img, (0,0),fx=0.5, fy=0.5)

cv2.imwrite(r"C:\Users\WTSCHGDEIS\Desktop"+"\\"+ModelName+str(n)+".jpg", img)
cv2.imshow("Wo isch de Walti?",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
