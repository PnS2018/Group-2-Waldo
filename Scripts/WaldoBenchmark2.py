#This script will use a pretrained Model to find Waldo.
#The best Matches are shown in red, and all other matches over the threshold will be in blue to black where blue is higher and black lower confidence.

import cv2
import numpy as np
import time
from keras.models import load_model
start = time.time()


#Directories and ModelNames. Model has to be in the same folder as this script
DataDir = r"F:\WaldoBilder\6.jpg"
ModelName = "waldoNewDataset.hdf5"

#Imagesize required by your model
ImgSize = 64

#stepsize of the search window
stepsize = 32

#Threshold for searching Algorithm
thresh = 0.9

#Number of top matches seen in red
max = 10

#pre-scaling factor
psf = 0.9

#load model
model = load_model(ModelName)

#preprocess image
img = cv2.imread(DataDir, 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (0, 0), fx=psf, fy=psf, interpolation=cv2.INTER_CUBIC)
a = img.shape



xit = int((a[0]-ImgSize)/stepsize)
yit = int((a[1]-ImgSize)/stepsize)
pred = np.ones((xit,yit), dtype='float32')

for x in range(xit):
    for y in range(yit):
        pic1 = img[x*stepsize:x*stepsize+ImgSize,y*stepsize:y*stepsize+ImgSize]
        pic = np.array([pic1])
        pred[x,y]= model.predict(pic/255.)
        print (model.predict(pic/255.))


 
z = 0
while pred[np.unravel_index(np.argmax(pred, axis=None), pred.shape)]>thresh:
    index = np.unravel_index(np.argmax(pred, axis=None), pred.shape)
    indexarr = np.asarray(index)
    
    print (pred[index])
    pred[index]= 0.
    
    pt1 = indexarr*stepsize
    pt2 = indexarr*stepsize+[ImgSize,ImgSize]
    pt1 = (pt1[1],pt1[0])
    pt2 = (pt2[1],pt2[0])
    if z < max:
        cv2.rectangle(img,pt1,pt2, (255,0,0),5)
    else:
        cv2.rectangle(img,pt1,pt2, (0,0,255-z,255-z),5)
    z = z+1
print ("Number of matches found: "+ str(z))
end = time.time()
print ("time elapsed: " + str(end-start)+" seconds")

scale = 0.5

img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Wo isch de Walti?", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
