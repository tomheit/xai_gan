import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from mnistCnn import MnistCnn
from mnistGan import MnistGan

#load images
mnist = tf.keras.datasets.mnist
(x_train, y_train_classes), (x_test, y_test_classes) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
x_train = x_train / 255
x_test = x_test / 255
#print(x_train.shape)
#load models
cnn = MnistCnn()
cnn.loadWeights('MnistCnnTEST')
gan = MnistGan()
gan.loadWeights('MnistGanTEST2/mnist_gen', 'MnistGanTEST2/mnist_disc')
#define explainer
@tf.function
def g(x, index, classifier, discriminator):
    res = discriminator(x) - tf.math.log(tf.squeeze(classifier(x))[index])
    return res

def explainer(x, classifier, discriminator, maxIter, maxChange, minAlpha):
    epsilon = 1e-16
    closeEnough = False
    iter = 0
    index = tf.argmax(tf.squeeze(classifier(x))).numpy()
    
    while(not closeEnough and iter < maxIter):
        with tf.GradientTape() as tape:
            gRes = g(x, index, classifier, discriminator)
        grad = tape.gradient(gRes, x)
        maxGrad = tf.abs(tf.reduce_max(grad))
        alpha = tf.minimum(minAlpha, maxChange/tf.maximum(maxGrad, epsilon))
        x.assign(x + alpha * grad)
        x.assign(tf.clip_by_value(x, clip_value_min = 0, clip_value_max = 1))
        iter += 1
        newIndex = tf.argmax(tf.squeeze(classifier(x))).numpy()
        if ((newIndex != index) and (discriminator(x).numpy() > 0)): # done when the prediction has changed
            closeEnough = True
    print("done after ", iter, " steps")
    return iter
#
#print(x_train[0].shape)
numImages = x_train.shape[0]
img_counter = 0
timeForSixtyImgs = time.time()
n600Steps = 0
_maxIter = 600
fig, ax = plt.subplots(1,2)
for img in x_train:
    start = time.time()
    #print(img.shape)
    x_ = tf.cast(np.expand_dims(img, 0), tf.float32)
    x = tf.Variable(x_)
    #print(x.shape)
    oldY = np.argmax(cnn.predict(x).numpy().squeeze())
    oldP = gan.discriminator(x).numpy()[0][0]
    n = explainer(x, cnn.model, gan.discriminator, maxIter = _maxIter, maxChange = 0.099, minAlpha = 1000000)
    newY = np.argmax(cnn.predict(x).numpy().squeeze())
    newP = gan.discriminator(x).numpy()[0][0]
    print("oldY: ", oldY, " oldP: ", oldP, " newY: ", newY, " newP: ", newP)
    if(n < _maxIter):
        title0 = "pred: " + str(oldY) + " p: " + str(oldP)
        title1 = "pred: " + str(newY) + " p: " + str(newP)
        #fig = plt.figure(figsize = (2,1))
        #_, ax = plt.subplots(1,2)
        ax[0].imshow(img, cmap = 'gray')
        ax[0].set_title(title0)
        ax[1].imshow(x[0], cmap = 'gray')
        ax[1].set_title(title1)
        plt.savefig('MnistExplainerImages/img_number_{:04d}.png'.format(img_counter))
        plt.cla()
    print(img_counter+1, "/", numImages, "(", time.time()-start, "s)")
    img_counter = img_counter + 1
    if n == 600:
        n600Steps = n600Steps + 1
    if img_counter > 59:
        break
print("Time for all images: ", time.time()-timeForSixtyImgs, "s")
print(n600Steps, "/", 60,)