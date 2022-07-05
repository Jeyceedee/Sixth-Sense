# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
from tensorflow import keras
from tensorflow.keras import activations, models, layers, callbacks
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

datasetPath = 'static\Dataset'
typeTumor = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
X_train = []
y_train = []
image_size = 150
for i in typeTumor:
    folderPath = os.path.join(datasetPath,'Training',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in typeTumor:
    folderPath = os.path.join(datasetPath,'Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, y_train = shuffle(X_train,y_train, random_state=101)
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
y_train_new = []

for i in y_train:
    y_train_new.append(typeTumor.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


y_test_new = []
for i in y_test:
    y_test_new.append(typeTumor.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

def create_model():
    conv_base = Xception(include_top=False, input_tensor=None,
    pooling=None, input_shape=(150, 150, 3), classifier_activation='softmax')
                               
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(4, activation = "softmax")(model)
    model = models.Model(conv_base.input, model)

    model.compile(optimizer = Adam(lr = 0.001),
                  loss = "categorical_crossentropy",
                  metrics = ["acc"])
    return model

model = create_model()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model_save = ModelCheckpoint('./Xception_best_weights2.h5', 
                              save_best_only = True, 
                              save_weights_only = True,
                              monitor = 'val_loss', 
                              mode = 'min', verbose = 1)
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                            patience = 5, mode = 'min', verbose = 1,
                            restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                               patience = 2, min_delta = 0.001, 
                               mode = 'min', verbose = 1) #reduced learning rate

"""
history  = model.fit(
                X_train,
                y_train,
                validation_split=0.2, 
                epochs =10, 
                verbose=1, 
                batch_size=32,
                callbacks=[model_save, early_stop, reduce_lr])
                """

model.load_weights('./Xception_best_weights2.h5')
"""
pred = model.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)
"""
"""
print(classification_report(y_test_new,pred,target_names = typeTumor))
"""

"""
def img_pred(upload):
    img = upload
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        print('The model predicts that there is no tumor')
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    if p!=1:
        print(f'The Model predicts that it is a {p}')

uploader = Image.open(r"static\Dataset\Testing\meningioma_tumor\image(127).jpg")
display(uploader)

img_pred(uploader)
"""