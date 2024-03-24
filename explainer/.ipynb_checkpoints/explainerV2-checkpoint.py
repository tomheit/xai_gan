# **** explainer class: performs gradient descent on an image to change the prediction ****
import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.nn import conv2d
from targetFunction import TargetFunction

class Explainer:
    def __init__(self, classifier, discriminator):
        self.classifier = classifier
        self.discriminator = discriminator
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
        self.err = 1e-16
    
    def explain(self, x, targetClass, targetProb = 0.99, distance = 'l1', l_H = 1.0, l_D = 1.0, l_d = 1.0, l_U = 1.0, maxIter = 500, epsilon = 0.01):
        """
            x: image to be explained
            targetClass: index of the desired class
            targetProb: probability of the desired class
            distance: distance function for regularization. one of l1, l2, l1_mad
            l_D: weight for the discriminator
            l_d: weight for the distance function
            l_U: weight for U(x)
            maxIter: maximum number of steps for gradient descent
        """
        # create target function
        targetFunction = TargetFunction(self.classifier, self.discriminator, x, distance, l_H, l_D, l_d, l_U)
        # unit vector of target class
        e_j = tf.reshape(tf.one_hot(indices = targetClass, depth = 10), shape = (1,10))
        e_j = tf.constant(e_j)
        # iteration counter
        iter = 0
        # end condition
        endCond = False
        # optimization loop
        while(not endCond and iter < maxIter):
            with tf.GradientTape() as tape:
                result = targetFunction.evaluate(x, e_j)
            grad = tape.gradient(result, x)
            x.assign(x - grad)
            x.assign(tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1))
            # increase iteration counter
            iter += 1
            # get new prediction
            prediction = tf.squeeze(self.classifier(x)).numpy()
            if(prediction[targetClass] >= targetProb):
                endCond = True
        return iter
