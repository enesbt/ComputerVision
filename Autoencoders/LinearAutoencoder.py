import  tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(X_train,y_train),(X_test,y_test)  = mnist.load_data()


X_train = X_train/255
X_test = X_test/255


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])


autoencoder = Sequential()

# Encode
autoencoder.add(Dense(128,activation='relu',input_dim=784))
autoencoder.add(Dense(64,activation='relu'))
autoencoder.add(Dense(32,activation='relu')) # Encoded image

# Decode
autoencoder.add(Dense(64,activation='relu'))
autoencoder.add(Dense(128,activation='relu'))
autoencoder.add(Dense(784,activation='sigmoid'))

#autoencoder.summary()

autoencoder.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
autoencoder.fit(X_train,X_train,epochs=50)

#encoder
encoder = Model(inputs = autoencoder.input,outputs=autoencoder.get_layer('dense_2').output)
plt.imshow(X_test[0].reshape(28,28),cmap='gray')

encoded_image = encoder.predict(X_test[0].reshape(1,-1))

#decoder
input_layer_decoder = Input(shape=(32,))
decoder_layer1 = autoencoder.layers[3]
decoder_layer2 = autoencoder.layers[4]
decoder_layer3 = autoencoder.layers[5]
decoder = Model(inputs=input_layer_decoder,outputs=decoder_layer3(decoder_layer2(decoder_layer1(input_layer_decoder))))
decoder.summary()
decoded_image = decoder.predict(encoded_image)
plt.imshow(decoded_image.reshape(28,28),cmap='gray')



n_images = 10
test_images = np.random.randint(0,X_test.shape[0]-1,size=n_images)

plt.figure(figsize=(18,18))

for i,image_index in enumerate(test_images):
    ax = plt.subplot(10,10,i+1)
    plt.imshow(X_test[image_index].reshape(28,28),cmap='gray')
    plt.xticks(())
    plt.yticks(())

    ax = plt.subplot(10,10,i+1+n_images)
    encoded_image = encoder.predict(X_test[image_index].reshape(1,-1))
    plt.imshow(encoded_image.reshape(8,4),cmap='gray')
    plt.xticks(())
    plt.yticks(())

    ax = plt.subplot(10,10,i+1+n_images*2)
    plt.imshow(decoder.predict(encoded_image).reshape(28,28),cmap='gray')
    plt.xticks(())
    plt.yticks(())


