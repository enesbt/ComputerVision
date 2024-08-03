import  tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPooling2D,UpSampling2D,Reshape,Flatten
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist,fashion_mnist

(X_train,y_train),(X_test,y_test)  = fashion_mnist.load_data()


X_train.shape,y_train.shape
classes =['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
width = 10
height = 10

fig,axes = plt.subplots(height,width,figsize=(15,15))
axes = axes.ravel()



for i in np.arange(0,width*height):
    index = np.random.randint(0,60000)
    axes[i].imshow(X_train[index],cmap='gray')
    axes[i].set_title(classes[y_train[index]],fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)


