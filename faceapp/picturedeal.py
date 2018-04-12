# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:31:32 2018

@author: ASUS
"""

import tensorflow as tf  
from scipy import misc  
import numpy as np  
import os
#随机旋转图片  
def random_rotate_image(image_file,hand, num):  
    with tf.Graph().as_default():  
        tf.set_random_seed(666)  
        file_contents = tf.read_file(image_file)  
        image = tf.image.decode_image(file_contents, channels=3)  
        image_rotate_en_list = []  
        def random_rotate_image_func(image):  
            #旋转角度范围  
            angle = np.random.uniform(low=-30.0, high=30.0)  
            return misc.imrotate(image, angle, 'bicubic')  
        for i in range(num):  
            image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8)  
            image_rotate_en_list.append(tf.image.encode_png(image_rotate))  
        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())  
            sess.run(tf.local_variables_initializer())  
            results = sess.run(image_rotate_en_list)  
            for idx,re in enumerate(results):  
                b=hand+400
                with open('D:/BaiduYunDownload/Machine Learning/Python27/faceRecognition/my_faces/99/'+str(b+idx)+'.jpg','wb') as f:  
                    f.write(re)  


#随机左右翻转图片  
def random_flip_image(image_file, hand,num):  
    with tf.Graph().as_default():  
        tf.set_random_seed(666)  
        file_contents = tf.read_file(image_file)  
        image = tf.image.decode_image(file_contents, channels=3)  
        image_flip_en_list = []  
        for i in range(num):  
            image_flip = tf.image.random_flip_left_right(image)  
            image_flip_en_list.append(tf.image.encode_png(image_flip))  
        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())  
            sess.run(tf.local_variables_initializer())  
            results = sess.run(image_flip_en_list)  
            for idx,re in enumerate(results):  
                b=hand+800
                with open('D:/BaiduYunDownload/Machine Learning/Python27/faceRecognition/my_faces/50/'+str(b+idx)+'.jpg','wb') as f:  
                    f.write(re)  
   
#随机裁剪图片  
def random_crop_image(image_file,hand, num):  
    with tf.Graph().as_default():  
        tf.set_random_seed(666)  
        file_contents = tf.read_file(image_file)  
        image = tf.image.decode_image(file_contents, channels=3)  
        image_crop_en_list = []  
        for i in range(num):  
            #裁剪后图片分辨率保持160x160,3通道  
            image_crop = tf.random_crop(image, [32, 32, 3])  
            image_crop_en_list.append(tf.image.encode_png(image_crop))  
        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())  
            sess.run(tf.local_variables_initializer())  
            results = sess.run(image_crop_en_list)  
            for idx,re in enumerate(results):  
                b=hand+1200
                with open('D:/BaiduYunDownload/Machine Learning/Python27/faceRecognition/my_faces/99/'+str(b+idx)+'.jpg','wb') as f:  
                    f.write(re)  


#加噪声
import cv2
def addnoise(img,hand):
    coutn=500
    for k in range(0,coutn):
        xi=int(np.random.uniform(0,img.shape[1]))#在0~shape[1]间取一随机数且强制转换为int类型
        xj=int(np.random.uniform(0,img.shape[0]))#在0~shape[0]间取一随机数且强制转换为int类型
        #下面三行是对RBG矩阵的R,B,G赋值
        img[xj,xi,0]=255*np.random.rand()#np.random.rand()取0~1的一个随机数
        img[xj,xi,1]=255*np.random.rand()
        img[xj,xi,2]=255*np.random.rand()
    b=hand+1600
    filename='D:/BaiduYunDownload/Machine Learning/Python27/faceRecognition/my_faces/99/'+str(b)+'.jpg'
    cv2.imwrite(filename,img) 



path='D:/BaiduYunDownload/Machine Learning/Python27/faceRecognition/my_faces/99'
i=1
for filename in os.listdir(path):
    if i>400:
        break
    else:
        filename = path+'/' + str(i)+'.jpg'
        print (filename)
        img=cv2.imread(filename)
        addnoise(img,i)  
    i=i+1
    
