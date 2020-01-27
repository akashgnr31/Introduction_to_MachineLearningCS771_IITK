import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model
from helpers import resize_to_fit
import time as tm

def decaptcha( filenames ):
    numChars = []
    codes = []
    m=0
    preprocessing=0
    testing=0
    tic = tm.perf_counter()
    model = load_model('final_model.hdf5')
    toc = tm.perf_counter()
    print(toc-tic)
    print(type(model))

    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    j = 0

    for i in filenames:
        tic=tm.perf_counter()

        img=cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        img[img==img[0,0]]=255
        kernel = np.ones((7,7), np.uint8) 
        kernel1 = np.ones((4,4), np.uint8) 
        img_d1 = cv2.dilate(img, kernel, iterations=1)
        img=cv2.cvtColor(img_d1,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(th3)
        if j==0:
            cv2.imshow("imagee",img)
            cv2.waitKey(0)
            cv2.destroyAllWindow()
        j = j+1

        toc3 = tm.perf_counter()

        cols = np.sum(img, axis=0)
        rows = np.sum(img, axis=1)
        cnt = 0
        flag = 0
        thres = 255*5
        tmp = 0
        start = []
        end = []
        for i in range(len(cols)):
            if cols[i] > thres and flag==0:
                flag = 1
                tmp = i
            if cols[i] <= thres and flag==1:
                if (i-tmp)>5 :
                    start.append(tmp)
                    end.append(i)
                flag = 0
        char = np.zeros((len(start),140,140))

        toc4 = tm.perf_counter()

        for i in range(len(start)):
            l = end[i] - start[i]
            char[i][:,75-(l//2):75+l-(l//2)] = img[5:-5,start[i]:end[i]]

        toc5 = tm.perf_counter()

        resized=np.zeros((len(start),30,30))
        for i in range(len(start)):
            char[i] = 255*np.ones((140,140)) - char[i]
            resized[i]=resize_to_fit(char[i], 30, 30)

        toc1=tm.perf_counter()

        temp1 = temp1 + (toc3 - tic)
        temp2 = temp2 + (toc4 - toc3)
        temp3 = temp3 + (toc5 - toc4)
        temp4 = temp4 + (toc1 - toc5)

        preprocessing=preprocessing+(toc1-tic)
            
        resized=resized[...,np.newaxis]
        y_out=model.predict(resized)
        label =""
        for i in range(len(start)):
            label=label+chr(ord('A')+np.where(y_out[i]==np.amax(y_out[i]))[0][0])
        numChars.append(len(start))
        codes.append(label)
        m=m+1

        toc2=tm.perf_counter()
        testing=testing+(toc2-toc1)


    print(temp1)
    print(temp2)
    print(temp3)
    print(temp4)        
    print(preprocessing)
    print(testing)
    
    return (numChars,codes)
   
   
