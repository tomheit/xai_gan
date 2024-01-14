import tensorflow as tf
import numpy as np

class Cgan_Explainer:
    
    def __init__(self):
        pass
    
    @tf.function
    def first_g(self, x, index, classifier):
        res = -tf.math.log(tf.squeeze(classifier(x))[index])
        return res

    @tf.function
    def second_g(self, x, index_np, index_tf, classifier, discriminator):
        res = tf.math.log(tf.squeeze(classifier(x))[index_np]) + 16*tf.math.log(discriminator([x, index_tf]))
        return res

    def explain(self, x, classifier, discriminator, maxIter, maxChange, minAlpha, discWeight):
        epsilon = 1e-16
        closeEnough = False
        iter = 0
        index = tf.argmax(tf.squeeze(classifier(x))).numpy()
        newIndex = index

        #first while loop min y_old
        while(not closeEnough and iter < maxIter):
            with tf.GradientTape() as tape:
                gRes = self.first_g(x, index, classifier)
            grad = tape.gradient(gRes, x)
            maxGrad = tf.abs(tf.reduce_max(grad))
            alpha = tf.minimum(minAlpha, maxChange/tf.maximum(maxGrad, epsilon))
            x.assign(x + alpha * grad)
            x.assign(tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1))
            iter += 1
            newIndex = tf.argmax(tf.squeeze(classifier(x))).numpy()
            if ((newIndex != index)): # done when the prediction has changed
                closeEnough = True
        print("first loop done after ", iter, " steps")

        newIndex_tf = tf.constant(np.expand_dims(newIndex, 0))


        #second while loop max y_new
        closeEnough = False
        iter = 0
        target_value = 0.9 #target value for y_new
        while(not closeEnough and iter < maxIter):
            with tf.GradientTape() as tape:
                hRes = self.second_g(x, newIndex, newIndex_tf, classifier, discriminator)
            grad = tape.gradient(hRes, x)
            maxGrad = tf.abs(tf.reduce_max(grad))
            alpha = tf.minimum(minAlpha, maxChange/tf.maximum(maxGrad, epsilon))
            x.assign(x + alpha * grad)
            x.assign(tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1))
            iter += 1
            y_new = tf.squeeze(classifier(x))[newIndex]
            if((y_new >= target_value) and (discriminator([x, newIndex_tf]).numpy() > 0.9)):
                closeEnough = True
                print("y_new: ", y_new)
        print("second loop done after ", iter, " steps")
        return iter, newIndex