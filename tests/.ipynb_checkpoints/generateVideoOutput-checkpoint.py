import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time

sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_gan')
sys.path.append('../explainer')

import explainer
import mnistCnn
import mnistGan

def main(argv):
    (X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_data = X_train.copy()
    img_data = img_data.reshape(X_train.shape[0], 28, 28, 1)
    img_data = img_data / 255 #pixel values in [0,1]
    #print(img_data.shape)
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    #print(test_data.shape)
    y_test_one_hot = np.eye(10)[y_test]
    
    numberOfExamples = 5
    epsilon = 0.01
    
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/largerCnn30Epochs')
    
    #load gan
    genPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_gen'
    discPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_disc'
    gan = mnistGan.MnistGan()
    gan.loadWeights(genPath, discPath)
    
    directory = './videosAttempt2/euclideanNorm'
    
    normC = 'euclidean'
    
    validTargetFunctions = ['loss', 'loss_gan', 'loss_gan_u', 'loss_u',
                            'negative_prob', 'negative_prob_gan', 'negative_prob_gan_u', 'negative_prob_u']
    
    # random images from the test data
    indices = np.random.randint(0, test_data.shape[0], numberOfExamples)
    # create explainer
    exp = explainer.Explainer(cnn.model, gan.discriminator)
    # predictions of the randomly selected elements
    predictionsOfIndices = tf.argmax(cnn.model(test_data[indices]), axis = 1).numpy()
    predictionsProb = cnn.model(test_data) #predictions of whole test data set
    predictions = tf.argmax(predictionsProb, 1) #classes of all predictions
    # confusion matrix
    confusionMatrix = tf.math.confusion_matrix(labels = y_test, predictions = predictions, num_classes = 10)
    
    fig = plt.figure()
    
    for targetF in validTargetFunctions:
        norm = colors.Normalize(vmin = 0, vmax = 1)
        targetDir = directory + "/" + targetF
        for i in range(numberOfExamples):
            targetDir = directory + "/" + targetF + "/vid_" + str(i)
            if(not os.path.exists(targetDir)):
                os.makedirs(targetDir)
            img = test_data[indices[i]]
            x = tf.Variable(tf.cast(np.expand_dims(img, 0), tf.float32))
            # select the second most commonly predicted class
            targetIndex = tf.math.top_k(confusionMatrix[predictionsOfIndices[i]], k=2).indices.numpy()[1]
            images = []
            outputs = []
            probabilities = []
            nIter, originalPred, originalConf, newPred, newConf = exp.explain(x, targetIndex, epsilon = epsilon, normConstraint = normC, targetFunction = targetF, genVideo = True, images = images, outputs = outputs, probabilities = probabilities)
            for j in range(len(images)):
                imgName = "frame_{:04d}.png".format(j)
                title = "class: " + str(outputs[j]) + " p: " + str(round(probabilities[j], 3))
                path = targetDir + "/" + imgName
                plt.imshow(images[j], cmap = 'gray', interpolation = 'none')
                fig.suptitle(title)
                plt.savefig(path)
                plt.cla()
    

if __name__ == "__main__":
    main(sys.argv[1:])