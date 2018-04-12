# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:00:23 2018

@author: ASUS
"""

import tensorflow as tf
import numpy as np
from faceapp.train import *
class A():
    def __init__(self):
        self.output = cnnlayer()  
        self.predict = tf.argmax(self.output, 1)
        self.sess = tf.Session()  
        self.saver = tf.train.Saver() 
        self.saver.restore(self.sess, tf.train.latest_checkpoint('faceapp/'))  
    
    def is_my_face(self,image):  
        res = self.sess.run(self.predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
        if res[0] == 1:  
            return 'ljf'  
        elif res[0]==0:
            return 'zr'
        elif res[0]==2:
            return 'lwc'
        elif res[0]==3:
            return 'ch'
        else:  
            return 'others' 
            
foo=A()
