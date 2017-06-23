#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:59:57 2017

@author: Shubham SIngh
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:59:59 2017

@author: Shubham Singh
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K 
K.set_image_dim_ordering('th') 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#%%
batch_size = 32
# number of output classes
nb_classes = 31
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


#%%
#  data

path1 = '/home/coe/objectdataset/inputdata'    #path of folder of images    
path2 = '/home/coe/objectdataset/inputdata_resized31'  #path of folder to save images    

listing = os.listdir(path2)
num_samples=size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '/' + file)  
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here          
    gray.save(path2 +'/' +  file, "JPEG")

imlist = sort(os.listdir(path2))

im1 = array(Image.open('/home/coe/objectdataset/inputdata_resized31' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('/home/coe/objectdataset/inputdata_resized31'+ '/' + im2)).flatten()
              for im2 in imlist],'f')
               
label=np.ones((num_samples,),dtype = int)
label[0:98]=0
label[98:195]=1 
label[195:346]=2
label[346:473]=3   
label[473:621]=4 
label[621:711]=5
label[711:817]=6  
label[817:1049]=7 
label[1049:1151]=8 
label[1151:1245]=9  
label[1245:1523]=10   
label[1523:1739]=11 
label[1739:1837]=12  
label[1837:1923]=13  
label[1923:2045]=14 
label[2045:2136]=15  
label[2136:2240]=16  
label[2240:2341]=17 
label[2341:2465]=18   
label[2465:2548]=19
label[2548:2690]=20   
label[2690:2787]=21 
label[2787:2897]=22 
label[2897:3009]=23 
label[3009:3123]=24 
label[3123:3229]=25 
label[3229:3329]=26 
label[3329:3439]=27 
label[3439:3641]=28 
label[3641:3843]=29
label[3843:3843]=30

names=['Ak47','American-flag','Bagpack','Baseball-bat','Baseball-glove','basketball-hoop','bat','bathtub','bear','beer-mug','billiards','binoculars','birdbath','blimp','bonsai','boombox','bowling-ball','bowling-pin','boxing-glove','brain101','breadmaker','buddha101','bulldozer','butterfly','cactus','cake','calculator','camel','Cat','Dog','Horse']


#Convert class labels to one-hot encoding
Y=np_utils.to_categorical(label,nb_classes)     


data,Label = shuffle(immatrix,label, random_state=3)
train_data = [data,Label]

img=data[196].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[1].shape)
print (train_data[0].shape)


#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 221
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

#%%

model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.05))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])

#%%

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch
               ,verbose=1, validation_data=(X_test, Y_test))

           
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10,
               verbose=1, validation_split=0.2)


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])




#%%      

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])



print(model.predict_classes(X_test[94:100]))
print(Y_test[94:100])



#%%

# visualizing intermediate layers

#output_layer = model.layers[1].get_output()
output_fn = theano.function([model.layers[0].get_input()], [model.layers[1].output()])

# the input image

input_image=X_train[0:1,:,:,:]
print(input_image.shape)

plt.imshow(input_image[0,0,:,:],cmap ='gray')
plt.imshow(input_image[0,0,:,:])


output_image = output_fn(input_image)
print(output_image.shape)

# Rearrange dimension so we can plot the result 
output_image = np.rollaxis(np.rollaxis(output_image, 3, 1), 3, 1)
print(output_image.shape)


fig=plt.figure(figsize=(8,8))
for i in range(32):
    ax = fig.add_subplot(6, 6, i+1)
    #ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
    ax.imshow(output_image[0,:,:,i],cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt

# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)



# Loading weights

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)
