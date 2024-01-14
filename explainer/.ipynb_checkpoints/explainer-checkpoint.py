import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.nn import conv2d

class Explainer:
    
    def __init__(self, model, discriminator):
        self.model = model
        self.discriminator = discriminator
        self.targetFuncStrings = ['loss', 'loss_gan', 'loss_gan_u', 'loss_u',
                                  'negative_prob', 'negative_prob_gan', 'negative_prob_gan_u', 'negative_prob_u',
                                  'prob', 'prob_gan']
        self.normStrings = ['max', 'euclidean', 'one']
        self.targetFunctions = {'loss':self.targetFuncLoss,
                                'loss_gan':self.targetFuncLossGan,
                                'loss_gan_u':self.targetFuncLossGanU,
                                'loss_u':self.targetFuncLossU,
                                'negative_prob':self.targetFuncNegativeProb,
                                'negative_prob_gan':self.targetFuncNegativeProbGan,
                                'negative_prob_gan_u':self.targetFuncNegativeProbGanU,
                                'negative_prob_u':self.targetFuncNegativeProbU,
                                'prob':self.targetFuncProb,
                                'prob_gan':self.targetFuncProbGan}
        self.normConstraints = {'max':self.maxNorm,
                                'euclidean':self.euclidNorm,
                                'one':self.oneNorm}
        self.cnnLoss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
        
    # Unglaette, sum of differences between pixels, horizontally and vertically (not diagonally)
    def computeU(self,img):
        kernel1 = np.array([[-1,1]], np.float32)
        kernel1 = np.expand_dims(kernel1, -1)
        kernel1 = np.expand_dims(kernel1, -1)
        kernel2 = np.array([[-1],[1]], np.float32)
        kernel2 = np.expand_dims(kernel2, -1)
        kernel2 = np.expand_dims(kernel2, -1)
        horizU = conv2d(img, kernel1, strides = (1,1,1,1), padding = 'VALID')
        vertU = conv2d(img, kernel2, strides = (1,1,1,1), padding = 'VALID')
        horizU = tf.abs(horizU)
        vertU = tf.abs(vertU)
        hSum = tf.reduce_sum(horizU)
        vSum = tf.reduce_sum(vertU)
        return hSum + vSum
        
    # target functions
    # loss
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncLoss(self, x, index, sign):
        # compute the loss of the prediction and a unit vector
        unitVector = tf.reshape(tf.one_hot(indices = index, depth = 10), shape = (1,10))
        res = sign * self.cnnLoss(self.model(x), unitVector)
        return res

    # loss + discriminator
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncLossGan(self, x, index, sign):
        unitVector = tf.reshape(tf.one_hot(indices = index, depth = 10), shape = (1,10))
        res = sign * self.cnnLoss(self.model(x), unitVector) + self.discriminator(x)
        return res
    
    # loss + discriminator - u
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncLossGanU(self, x, index, sign):
        unitVector = tf.reshape(tf.one_hot(indices = index, depth = 10), shape = (1,10))
        res = sign * self.cnnLoss(self.model(x), unitVector) + self.discriminator(x) - self.computeU(x)
        return res
    
    # loss - u
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncLossU(self, x, index, sign):
        # compute the loss of the prediction and a unit vector
        unitVector = tf.reshape(tf.one_hot(indices = index, depth = 10), shape = (1,10))
        res = sign * self.cnnLoss(self.model(x), unitVector) - self.computeU(x)
        return res

    # negative probability to be minimized
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncNegativeProb(self, x, index, sign):
        # compute the natural logarithm of the predicted probability
        res = sign * (-tf.math.log(tf.squeeze(self.model(x))[index]))
        return res

    # negative prob + discriminator
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncNegativeProbGan(self, x, index, sign):
        res = sign * (-tf.math.log(tf.squeeze(self.model(x))[index])) + self.discriminator(x)
        return res
    
    # negative prob + discriminator - u
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncNegativeProbGanU(self, x, index, sign):
        res = sign * (-tf.math.log(tf.squeeze(self.model(x))[index])) + self.discriminator(x) - self.computeU(x)
        return res
    
    # negative prob - u
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncNegativeProbU(self, x, index, sign):
        # compute the natural logarithm of the predicted probability
        res = sign * (-tf.math.log(tf.squeeze(self.model(x))[index])) - self.computeU(x)
        return res

    # positive probability to be maximized
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncProb(self, x, index, sign):
        res = sign * tf.math.log(tf.squeeze(self.model(x))[index])
        return res

    # positive prob + discriminator
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec(None, dtype = tf.int64), tf.TensorSpec(None, dtype = tf.float32)))
    def targetFuncProbGan(self, x, index, sign):
        res = sign * tf.math.log(tf.squeeze(self.model(x))[index]) + self.discriminator(x)
        return res
    
    # norm constraints
    # maximum norm
    def maxNorm(self, grad, epsilon, x):
        res = epsilon * tf.math.sign(grad)
        return res

    # euclidean norm
    def euclidNorm(self, grad, epsilon, x):
        err = 1e-16
        d = 28*28
        norm = tf.norm(grad, ord='euclidean')
        res = epsilon*math.sqrt(d)*(grad/tf.maximum(norm, err))
        return res
    
    # 1-norm
    # maximize / minimize the pixels with the greatest impact on the prediction
    # one step maximally perturbs pixels with the greatest impact with a maximum perturbation of epsilon
    def oneNorm(self, grad, epsilon, x):
        #to do: optimize this code
        d = 28*28
        abs_grad = tf.abs(grad)
        sign_grad = tf.math.sign(grad)
        budget = epsilon * d
        # get indices of grad sorted by greatest abs value as 1xd-vector
        abs_grad_sorted_indices = tf.argsort(tf.reshape(abs_grad, shape = (d,)), direction='DESCENDING')
        # x = x + eta
        eta = tf.zeros_like(grad)
        #print(eta.shape)
        for i in range(d):
            index = abs_grad_sorted_indices[i] #start iterating over indices
            row, col = self.vecIndexToMatIndex(index = index, dim = 28) #corresponding row and column
            #get diff to 0 or 1 depending on the sign of the gradient:
            diff = sign_grad[0,row,col,0] * tf.abs(x[0,row,col,0] - ((sign_grad[0,row,col,0] / 2) + 0.5))
            if(budget >= tf.abs(diff)):
                #eta[0,row,col,0] += diff
                eta = eta + np.expand_dims(np.expand_dims((diff * self.getUnitMat(eta.shape[1],row,col)),0),3)
                budget -= tf.abs(diff)
            else:
                #eta[0,row,col,0] += (sign_grad[0,row,col,0] * budget)
                eta = eta + np.expand_dims(np.expand_dims(((sign_grad[0,row,col,0] * budget) * self.getUnitMat(eta.shape[1],row,col)),0),3)
                budget = 0
            if(budget <= 0):
                break
        return eta
    
    # utility functions
    # utility function to get row and col from an index
    def vecIndexToMatIndex(self, index, dim):
        row = math.floor(index/dim)
        col = index % dim
        return row, col

    # utility function to get a unit matrix
    def getUnitMat(self, dim, row, col):
        # dim: number of rows/columns
        # row: index of row
        # col: index of column
        if(row >= dim):
            raise ValueError("row index out of bounds")
        if(col >= dim):
            raise ValueError("column index out of bounds")
        index = row*(dim) + col
        e = tf.one_hot(indices = index, depth = dim*dim)
        mat = tf.reshape(e, shape = (dim,dim))
        return mat
    
    # end conditions
    def endConditionNew(self, index, targetIndex, newIndex):
        return (index != newIndex)
    
    def endConditionTarget(self, index, targetIndex, newIndex):
        return (index == targetIndex)
    
    # explainer
    def explain(self, x, targetIndex = -1, maxIter = 600, epsilon = 0.1, err = 1e-16, normConstraint = 'euclidean', targetFunction = 'negativeProb'):
        # check if a valid target function and norm constraint were given
        if(targetFunction not in self.targetFuncStrings):
            raise ValueError("target function must be in " f"{self.targetFuncStrings}, got {targetFunction}")
        if(normConstraint not in self.normStrings):
            raise ValueError("norm constraint must be in " f"{self.normStrings}, got {normConstraint}")
            
        # convert strings to actual target functions
        targetFunc = self.targetFunctions[targetFunction]
        normConst = self.normConstraints[normConstraint]
        
        # get the original output
        originalOutput = tf.squeeze(self.model(x))
            
        # define constants
        closeEnough = False
        iter = 0
        index = tf.argmax(originalOutput).numpy() # originally predicted class
        indexAsTensor = tf.constant(targetIndex, tf.int64) # target index as tensor
        newIndex = index
        d = 28*28
        
        # define return values
        originalPred = index
        originalConf = originalOutput[index].numpy()
        
        # set end condition
        sign = -1
        endCondition = self.endConditionTarget
        if(targetIndex == -1):
            endCondition = self.endConditionNew
            sign = 1
            indexAsTensor = tf.constant(index, tf.int64) # current index as tensor
        signAsTensor = tf.constant(sign, tf.float32)
        
        # optimization loop
        while(not closeEnough and iter < maxIter):
            with tf.GradientTape() as tape:
                res = targetFunc(x, indexAsTensor, signAsTensor)
            grad = tape.gradient(res, x)
            eta = normConst(grad, epsilon, x)
            x.assign(x + eta)
            x.assign(tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1))
            iter += 1
            newIndex = tf.argmax(tf.squeeze(self.model(x))).numpy()
            if(endCondition(index, targetIndex, newIndex)):
                closeEnough = True
                
        # define return values
        newOutput = tf.squeeze(self.model(x))
        newPred = tf.argmax(newOutput).numpy()
        newConf = newOutput[newIndex].numpy()
        
        return iter, originalPred, originalConf, newPred, newConf