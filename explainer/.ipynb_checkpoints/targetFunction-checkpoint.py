# **** constructs the target function that is used in gradient descent ****
import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.nn import conv2d

class TargetFunction:
    def __init__(self, cnn, discriminator, x_old, distance, l_H = 1.0, l_D = 1.0, l_d = 1.0, l_U = 1.0):
        self.cnn = cnn
        self.Discriminator = discriminator
        self.x_old = tf.identity(x_old)
        self.MAD = self.getMAD()
        self.distance = self.getDistance(distance)
        self.crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
        self.l_H = l_H
        self.l_D = l_D
        self.l_d = l_d
        self.l_U = l_U
        self.loss = lambda e_j, x : self.l_H * self.crossentropy(e_j, self.cnn(x))
        self.disc = lambda x : (self.l_D * self.Discriminator(x) if l_D > 0.0 else 0.0)
        self.dist = lambda x : (self.l_d * self.distance(x, self.x_old) if l_d > 0.0 else 0.0)
        self.u = lambda x : (self.l_U * self.U(x) if l_U > 0.0 else 0.0)
    
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec([None,None], dtype = tf.float32)))
    def evaluate(self, x, e_j):
        loss = self.loss(e_j, x)
        discriminator = self.disc(x)
        distance = self.dist(x)
        u = self.u(x)
        return loss - discriminator + distance + u
    
    def U(self, x):
        height = x.shape[1]
        width = x.shape[2]
        # factor for normalization:
        factor = 1/(height*(width-1)+width*(height-1))
        kernel1 = np.array([[-1,1]], np.float32)
        kernel1 = np.expand_dims(kernel1, -1)
        kernel1 = np.expand_dims(kernel1, -1)
        kernel2 = np.array([[-1],[1]], np.float32)
        kernel2 = np.expand_dims(kernel2, -1)
        kernel2 = np.expand_dims(kernel2, -1)
        horizU = conv2d(x, kernel1, strides = (1,1,1,1), padding = 'VALID')
        vertU = conv2d(x, kernel2, strides = (1,1,1,1), padding = 'VALID')
        horizU = tf.abs(horizU)
        vertU = tf.abs(vertU)
        hSum = tf.reduce_sum(horizU)
        vSum = tf.reduce_sum(vertU)
        result = factor*(hSum + vSum) #normalize U
        return result
    
    def L1Dist(self, x, z):
        return tf.reduce_sum(tf.math.abs(x - z))
    
    def L2Dist(self, x, z):
        err = (1/(255*255)) # add err to avoid nan
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(x-z)) + err)
    
    def MADL1Dist(self, x, z):
        diff = tf.math.abs(x - z)
        divide = tf.math.divide(diff, self.MAD)
        return tf.math.reduce_sum(divide)
    
    def getMAD(self):
        # load mnist dataset
        (X_train, _), _ = tf.keras.datasets.mnist.load_data()
        train_data = X_train.copy()
        train_data = train_data.reshape(X_train.shape[0], 28, 28, 1)
        train_data = train_data / 255 #pixel values in [0,1]
        # compute median
        x_median = np.median(train_data, axis=0)
        # compute median absolute distance
        MAD = np.median(np.abs(train_data - x_median), axis = 0)
        # convert to tensor
        MAD_tf = tf.constant(tf.cast(np.expand_dims(MAD, 0), tf.float32))
        # to avoid dividing by 0 add small err where MAD is 0.0
        # using a very small err (like 1e-16) will lead to exploding gradients
        err = 1 / 255
        mask = tf.equal(MAD_tf, 0.0)
        MAD_tf_err = tf.where(mask, tf.fill(MAD_tf.shape, err), MAD_tf)
        return MAD_tf_err
    
    def getDistance(self, distance):
        distances = {'l1':self.L1Dist,
                     'l2':self.L2Dist,
                     'l1_mad':self.MADL1Dist}
        returnDist = distances['l1']
        if(distance in distances):
            returnDist = distances[distance]
        return returnDist