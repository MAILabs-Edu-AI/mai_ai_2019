import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Input, Dense 
from keras.utils import np_utils 

DATADIR = "/home/den/n/data"

CATEGORIES = ["X", "V"]

data=[]
class_list = []


for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
        data.append(img_array)
        class_list.append(category)


#print(class_list)
num_class_list=[]

for i in class_list:
    if i=='X':
        num_class_list.append(0)
    if i=='V':
         num_class_list.append(1)
#print(num_class_list)

num_class_arr=np.array(num_class_list)
#print(num_class_arr)
#print(type(num_class_arr))
data_arr=np.array(data)
#print(type(data_arr))
#print(len(data_arr))
#print(data_arr[0])

batch_size = 15 # количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска examples at once
num_epochs = 20 # количество итераций обучающего алгоритма по всему обучающему множеству
hidden_size = 100 #количество нейронов в каждом из двух скрытых слоев MLP

num_train = 320
num_test = 80

height, width, depth = 95, 95, 1 
num_classes = 2 


X_train=data_arr[0:320]
y_train=num_class_arr[0:320]


X_test=data_arr[320:400]
y_test=num_class_arr[320:400]

print(len(X_train))
print(X_train[0])

X_train = X_train.reshape(num_train, height * width) 
X_test = X_test.reshape(num_test, height * width) 

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
X_train /= 255 
X_test /= 255 

Y_train = np_utils.to_categorical(y_train, num_classes) 
Y_test = np_utils.to_categorical(y_test, num_classes) 

inp = Input(shape=(height * width,)) 
hidden_1 = Dense(hidden_size, activation='relu')(inp) # First hidden ReLU layer
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1) # Second hidden ReLU layer
out = Dense(num_classes, activation='softmax')(hidden_2) # Output softmax layer

model = Model(input=inp, output=out) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=num_epochs) 



