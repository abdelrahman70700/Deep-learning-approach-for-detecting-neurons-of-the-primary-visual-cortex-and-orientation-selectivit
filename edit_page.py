import cv2
from tifffile import imsave
import numpy as np
import random
import glob
import math
import statistics
import scipy
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import time
import hydralit_components as hc
#------------------------------------------ML---------------------------------------------------
import numpy as np
from skimage import io
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import zipfile

from websocket._http import proxy_info


def extract(file):
    with zipfile.ZipFile(file,"r") as zip_ref:
         zip_ref.extractall("extracted/")
    return zip_ref.namelist()[0]




def vidProcessing(Video):
    l = Video.shape[0]
    m = Video.shape[1]
    n = Video.shape[2]
    maxI = np.empty([m, n], dtype=np.single)
    minI = np.empty([m, n], dtype=np.single)
    for i in range(l):
        Video[i] = cv2.medianBlur(Video[i], 3)
    for i in range(m):
        for j in range(n):
            v = Video[:, i, j]
            maxI[i][j] = max(v)
            minI[i][j] = min(v)
    return Video, maxI, minI


def imgProcessing(maxI, minI, dim):
    maxI = cv2.resize(maxI, dim)
    maxIN = maxI - np.amin(maxI)
    maxIN = maxIN / np.amax(maxIN)
    minI = cv2.resize(minI, dim)
    minIN = minI - np.amin(minI)
    minIN = minIN / np.amax(minIN)
    return maxI, minI, maxIN, minIN


def Cutter(folder, Arr, m=126, n=126, d=9, s=3):
    c = 0
    for i in range(0, m - d + 1, s):
        for j in range(0, n - d + 1, s):
            cv2.imwrite(folder + f"//{c} {i} {j}.jpg", Arr[i:i + d, j:j + d])
            c += 1


def prepareLabels(path, folder):
    dim = (504, 504)
    img = cv2.imread(path)
    img = cv2.resize(img, dim)
    if not os.path.exists(folder):   os.mkdir(folder)
    Cutter(folder, img, dim[0], dim[1], 36, 12)  # 504 126


def Cut(Mini, Maxi, m=126, n=126, d=9, s=3):
    l = []
    h = ((m - d) / s) + 1
    h = int(h * h)
    MiniMax = np.empty([h, d, d, 2], dtype=np.single)
    MiniMaxFlat = np.empty([h, 2, d * d], dtype=np.single)
    c = 0
    for i in range(0, m - d + 1, s):
        for j in range(0, n - d + 1, s):
            MiniMax[c, :, :, 0] = Mini[i:i + d, j:j + d] - 0.5
            MiniMaxFlat[c, 0, :] = Mini[i:i + d, j:j + d].flatten() - 0.5
            MiniMaxFlat[c, 1, :] = Maxi[i:i + d, j:j + d].flatten() - 0.5
            MiniMax[c, :, :, 1] = Maxi[i:i + d, j:j + d] - 0.5
            l.append((i, j))
            c += 1
    return MiniMax, MiniMaxFlat, l, h


def readLabels(path, st):
    y = [0] * st
    for image in os.listdir("New folder (3)"):
        y[int(image.split()[0])] = 1
    return np.array(y)


def graphing(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    ax = plt.gca()
    ax.set_ylim([0.4, 1])
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    ax = plt.gca()
    ax.set_ylim([0, 1.8])
    plt.show()


def model_1d(XF, y):
    model = models.Sequential()
    model.add(Bidirectional(layers.LSTM(24, activation='tanh'), input_shape=(2, 81)))
    model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.5)))
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy
                  , metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

    X_train, X_CV, y_train, y_CV = train_test_split(XF, y, test_size=0.2)
    history = model.fit(X_train, to_categorical(y_train), epochs=1080, batch_size=32,
                        validation_data=(X_CV, to_categorical(y_CV)))
    model.save('model1d.h5')

    graphing(history)


def model_2d(X, y):
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size=[3, 3], padding='same', activation='tanh', input_shape=(9, 9, 2)))
    model.add(layers.MaxPooling2D(pool_size=[2, 2]))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=[2, 2]))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.3)))
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy
                  , metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

    X_train, X_CV, y_train, y_CV = train_test_split(X, y, test_size=0.2)
    history = model.fit(X_train, to_categorical(y_train), epochs=330, batch_size=20,
                        validation_data=(X_CV, to_categorical(y_CV)))
    model.save('model2d.h5')

    graphing(history)


def model_3d(X, y):
    model = models.Sequential()
    model.add(layers.Conv3D(32, kernel_size=[3, 3, 1], activation='tanh', padding='same', input_shape=(9, 9, 2, 1)))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(64, kernel_size=[3, 3, 2], activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='sigmoid'))
    model.add(layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.07)))
    model.summary()

    model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

    X = np.expand_dims(X, axis=4)
    X_train, X_CV, y_train, y_CV = train_test_split(X, y, test_size=0.2)
    history = model.fit(X_train, to_categorical(y_train), epochs=330, batch_size=32,
                        validation_data=(X_CV, to_categorical(y_CV)))
    model.save('model3d.h5')

    graphing(history)


def predictor(model, X):
    p = model.predict(X)
    yp = p[:, 1] > p[:, 0]
    return yp


def predictors(X, XF, t=False):
    model = models.load_model('model1d.h5')
    yp = predictor(model, XF)
    modell = models.load_model('model2d.h5')
    ypp = predictor(modell, X)
    modelll = models.load_model('model3d.h5')
    yppp = predictor(modelll, np.expand_dims(X, axis=4))
    voting = yp + 1 - 1
    voting = (voting + ypp) + yppp
    votBool = voting > 1
    if t:   return yp, ypp, yppp, votBool, voting
    return votBool


def testing(X, XF, y):
    l = len(y)
    yp, ypp, yppp, votBool, voting = predictors(X, XF, True)
    print(f"The accuracy of 1d predictor = {np.mean(y == yp)} with {np.sum(y != yp)} errors out of {l} .")
    print(f"The accuracy of 2d predictor = {np.mean(y == ypp)} with {np.sum(y != ypp)} errors out of {l} .")
    print(f"The accuracy of 3d predictor = {np.mean(y == yppp)} with {np.sum(y != yppp)} errors out of {l} .")
    print(f"The accuracy of  homoPredictor = {np.mean(y == votBool)} with {np.sum(y != votBool)} errors out of {l} .")
    print("The testing errors on voting are :")
    for i in range(l):
        if y[i] != votBool[i]:
            print(i, "   ", votBool[i], "   ", voting[i])


def neuroSearch(votBool, loc, maxI):
    grid = {}
    for i in range(len(votBool)):
        a, b = loc[i]
        if votBool[i]:  grid[loc[i]] = np.mean(maxI[a:a + 9, b:b + 9])
    gridV = list(grid.keys())
    neurons = []
    for g in gridV:
        a = (g[0] + 3, g[1])
        b = (g[0], g[1] + 3)
        c = (g[0] + 3, g[1] + 3)
        d = (g[0] + 6, g[1] + 6)
        maxi = g
        if a in gridV:
            if grid[a] > grid[maxi]:  maxi = a
            gridV.remove(a)
        if b in gridV:
            if grid[b] > grid[maxi]:  maxi = b
            gridV.remove(b)
        if c in gridV:
            if grid[c] > grid[maxi]:  maxi = c
            if not d in gridV:  gridV.remove(c)
        neurons.append(maxi)
    return neurons


def neuroArt(img, neurons, m=1):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch
    # Add the patch to the Axes

    for i, x in enumerate(neurons):
        rect = patches.Rectangle((int(x[1] * m), int(x[0] * m)),
                                 int(9 * m), int(9 * m), linewidth=1, edgecolor='r', facecolor='none')
        # ax.annotate(f"{i}",(int((x[1]+1)*m), int((x[0]+7)*m)),fontsize=8)
        ax.add_patch(rect)
    col1,col2=st.columns([0.5,0.5])
    with col1:
        st.pyplot(fig,width=400)
        plt.close()

def neuroIntensity(Video, neurons, dim):
    avgIntensity = np.empty([Video.shape[0], len(neurons)])
    i = 0
    for img in Video:
        img = cv2.resize(img, dim)
        avgIntensity[i, :] = [np.mean(img[x[0]:x[0] + 9, x[1]:x[1] + 9]) for x in neurons]
        i += 1
    return avgIntensity


def reSize(maxI, minI):
    model = models.load_model('model3d.h5')
    pNeurons = []
    Dim = [(63, 63)]
    Dim += [(i * 126, i * 126) for i in range(1, 9)]
    for dim in Dim:
        a, _, maxIN, minIN = imgProcessing(maxI, minI, dim)
        X, XF, loc, st = Cut(minIN, maxIN, dim[0], dim[1])
        p = model.predict(np.expand_dims(X, axis=4))
        pp = p[:, 1]
        if len(pp[pp > 0.5]) == 0:
            pNeurons.append(0)
        else:
            pNeurons.append(len(pp[pp > 0.8]) / len(pp[pp > 0.5]))
        print(pNeurons)
    i = np.argmax(pNeurons)
    return Dim[i]


def apply(path):
    Video = io.imread(path)
    Video, maxI, minI = vidProcessing(Video)
    dim = (126, 126)
    maxI, minI, maxIN, minIN = imgProcessing(maxI, minI, dim)
    prepareLabels("Captured.PNG", "neuroNumber9")
    # putting manual labels on the images
    X, XF, loc, st = Cut(minIN, maxIN)
    y = readLabels("New folder (3)", st)
    model_1d(XF, y)
    model_2d(X, y)
    model_3d(X, y)
    testing(X, XF, y)
    yp = predictors(X, XF)
    neurons = neuroSearch(yp, loc, maxI)
    neuroArt(minI, neurons)
    neuroArt(maxI, neurons)
    neuroArt(cv2.resize(cv2.imread("Captured.PNG"), dim), neurons)


def application(path):
    Video = io.imread(path)
    Video, maxI, minI = vidProcessing(Video)
    dim = reSize(maxI, minI)
    a, _, maxIN, minIN = imgProcessing(maxI, minI, dim)
    X, XF, loc, st = Cut(minIN, maxIN, dim[0], dim[1])
    yp = predictors(X, XF)
    neurons = neuroSearch(yp, loc, a)
    avgIntensity = neuroIntensity(Video, neurons, dim)
    mult = maxI.shape[0] / dim[0]
    print(mult)
    neuroArt(minI, neurons, mult)
    neuroArt(maxI, neurons, mult)
    return avgIntensity


#---------------------------------------------------------------------------------------------


def edit_page():
    logo = Image.open('index.png')
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload your Video here...</p>', unsafe_allow_html=True)
        
    with col2:               # To display brand logo
        st.image(logo,  width=130)
    uploaded_file = st.file_uploader("", type=['zip','mp4'])
    if uploaded_file is not None:
        x=extract(uploaded_file)
        # print(extract(uploaded_file))
        # print(os.path(x))
        # print(x)
        # print(uploaded_file)
        split_tup = os.path.splitext(x)
        file_extension = split_tup[1]
        if file_extension=='.mp4':
            x=mp4Tif(x,"extracted/"+"video.tif")
            image=application(x)
        else:
            image = application("extracted/"+x)

        np.save("data",image)
        # print(image)
        # print(len(image))

        # st.image(image)
        # col1, col2 = st.columns( [0.5, 0.5])
        # with col1:
        #     st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        #     st.image(image,width=300)  

        # with col2:
        #     st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)

        #     converted_img = np.array(image.convert('RGB')) 
        #     gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        #     inv_gray = 255 - gray_scale
        #     blur_image = cv2.GaussianBlur(inv_gray, (125,125), 0, 0)
        #     sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
        #     st.image(sketch, width=300)




#---------------------------------------------------------------------------------------------
def photo():
    edit_page()
    resizedImages = []
    for img in glob.glob("images/*.jpg"):
        cv_img = plt.imread(img)
        resizedImages.append(cv_img)
    caption = [""]
    idx = 0
    for _ in range(len(resizedImages) - 1):
        cols = st.columns(3)

        if idx < len(resizedImages):
            cols[0].image(resizedImages[idx], width=250, caption=caption[0])
        idx += 1

        if idx < len(resizedImages):
            cols[1].image(resizedImages[idx], width=250, caption=caption[0])
        idx += 1

        if idx < len(resizedImages):
            cols[2].image(resizedImages[idx], width=250, caption=caption[0])
            idx += 1
        else:
            break



def mp4Tif(path, out):
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    Video = []
    count = 0

    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        Video.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        success, image = vidcap.read()
        print(count, 'Read a new frame: ', success)
        count += 1

    Video = np.array(Video)
    imsave(out, Video)
    return out

