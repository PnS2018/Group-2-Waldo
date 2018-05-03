#Script to visualize the Convolutional filters from the first layer

from keras.models import load_model
import matplotlib.pyplot as plt


model = load_model("WaldoS2.hdf5")
print('Model loaded.')

model.summary()

model1w = model.get_weights()[0]
print(model1w.shape)
for i in range(model1w.shape[3]):
    plt.subplot(5,10,i+1)
    filter = model1w[:,:, :, i]-model1w[:,:, :, i].min()
    filter = filter / filter.max()
    plt.imshow(filter,interpolation="nearest")
plt.show()