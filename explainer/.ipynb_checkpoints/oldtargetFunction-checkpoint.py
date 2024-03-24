import tensorflow as tf
# This class serves as the target function that is to be optimized
class TargetFunction:
    
    def __init__(self, classifier, targetIndex, objective = 'loss', distance = 'none', uValue = None, discriminator = None, MAD = None, wObj = 1, wDisc = 1, wU = 1, wDist = 1):
        self.cnnLoss = tf.keras.losses.CategoricalCrossentropy(from_logits = False)
        self.classifier = classifier
        self.discriminator = discriminator
        # set discriminator to const 0 if it is None
        self.setDiscriminator(discriminator)
        # set uValue to const 0 if it is None
        self.uValue = self.computeU
        self.setUValue(uValue)
        self.targetVector = self.setTargetVector(targetIndex)
        self.targetIndex = targetIndex
        self.MAD = MAD
        self.objectives = ['loss', 'probability']
        self.distances = ['none', 'l1', 'l1MAD']
        # define objective to be minimized/maximized
        # minimize loss of target or maximize target probability
        self.getObjective = {'loss':self.lossObj,
                             'probability':self.probObj}
        self.objective = self.getObjective[objective]
        # set distance to get adversaries near x
        self.getDistance = {'none':self.noDist,
                            'l1':self.distL1,
                            'l1MAD':self.distMAD}
        self.distance = self.getDistance[distance]
        # set weights
        self.wObj = wObj
        self.wDisc = wDisc
        self.wU = wU
        self.wDist = wDist
            
    def lossObj(self, x):
        return -self.cnnLoss(self.classifier(x), self.targetVector)
    
    def probObj(self, x):
        return tf.math.log(tf.squeeze(self.classifier(x)))[self.targetIndex]
    
    def noDist(self, x, xOld):
        return 0
    
    def distL1(self, x, xOld):
        diff = tf.math.abs(x - xOld)
        return tf.math.reduce_sum(diff)
    
    def distMAD(self, x, xOld):
        diff = tf.math.abs(x - xOld)
        result = tf.math.divide(diff, self.MAD)
        return tf.math.reduce_sum(result)
        
    def setTargetVector(self, targetIndex):
        targetVector = tf.reshape(tf.one_hot(indices = targetIndex, depth = 10), shape = (1,10))
        return targetVector
        
    def setDiscriminator(self, discriminator):
        if(discriminator is None):
            self.discriminator = self.zero
        else:
            self.discriminator = discriminator
            
    def setUValue(self, uValue):
        if(uValue is None):
            self.uValue = self.zero
            
    def computeU(self, img):
        height = img.shape[1]
        width = img.shape[2]
        # factor for normalization:
        factor = 1/(height*(width-1)+width*(height-1))
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
        result = factor*(hSum + vSum) #normalize U
        return result
            
    def zero(self, x):
        return 0
    
    @tf.function(input_signature=(tf.TensorSpec([None,None,None,None], dtype = tf.float32), tf.TensorSpec([None,None,None,None], dtype = tf.float32)))
    def apply(self, x, xOld):
        return self.wObj*self.objective(x) + self.wDisc*self.discriminator(x) - self.wU*self.uValue(x) - self.wDist*self.distance(x, xOld)