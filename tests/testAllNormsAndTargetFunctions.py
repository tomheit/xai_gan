import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_gan')
sys.path.append('../neural_networks/mnist_wgan')
sys.path.append('../explainer')

import explainer
import mnistCnn
import mnistGan
import wgan

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
    
    numberOfExamples = 50
    epsilon = 0.01
    
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/largerCnn30Epochs')
    
    #load gan (old gan model)
    #genPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_gen'
    #discPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_disc'
    #gan = mnistGan.MnistGan()
    #gan.loadWeights(genPath, discPath)
    genPath = '../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_generator.h5'
    discPath = '../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5'
    invPath = '../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_inverter.h5'
    wGan = wgan.WGan()
    wGan.load_gan(genPath, discPath, invPath)
    
    directory = "./normalizedU_e0_01_WGan/"
    
    normConstraints = ['euclidean', 'max', 'one']
    
    validTargetFunctions = ['loss', 'loss_gan', 'loss_gan_u', 'loss_u',
                            'negative_prob', 'negative_prob_gan', 'negative_prob_gan_u', 'negative_prob_u']
    
    classSelections = ['confusion', 'second']
    
     # random images from the test data
    indices = np.random.randint(0, test_data.shape[0], numberOfExamples)
    # create explainer
    exp = explainer.Explainer(cnn.model, wGan.discriminator)
    # predictions
    predictionsOfIndices = tf.argmax(cnn.model(test_data[indices]), axis = 1).numpy()
    predictionsProb = cnn.model(test_data)
    predictions = tf.argmax(predictionsProb, 1)
    confusionMatrix = tf.math.confusion_matrix(labels = y_test, predictions = predictions, num_classes = 10)
    
    fig, ax = plt.subplots(1,3)
    for targetClass in classSelections:
        for normC in normConstraints:
            print(normC)
            for targetF in validTargetFunctions:
                targetDir = directory + targetClass + "/" + normC + "Norm/" + targetF
                if(not os.path.exists(targetDir)):
                    os.makedirs(targetDir)
                for i in range(numberOfExamples):
                    title = "img"+str(i)
                    img = test_data[indices[i]]
                    x = tf.Variable(tf.cast(np.expand_dims(img, 0), tf.float32))
                    if(targetClass == 'confusion'):
                        # select the second most commonly predicted class
                        targetIndex = tf.math.top_k(confusionMatrix[predictionsOfIndices[i]], k=2).indices.numpy()[1]
                    else:
                        targetIndex = -1
                    nIter, originalPred, originalConf, newPred, newConf = exp.explain(x, targetIndex, epsilon = epsilon, normConstraint = normC, targetFunction = targetF)
                    title0 = "pred: " + str(originalPred) + ", p: " + str(originalConf)
                    title1 = "pred: " + str(newPred) + ", p: " + str(newConf)
                    ax[0].imshow(img, cmap = 'gray', interpolation = 'none')
                    ax[0].set_title(title0)
                    ax[1].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
                    ax[1].set_title(title1)
                    diff = ((x[0] - img)/2 + 0.5)
                    ax[2].imshow(diff, cmap = 'seismic', interpolation = 'none')
                    ax[2].set_title("diff")
                    plt.savefig(targetDir + "/img_{:04d}.png".format(i))
                    plt.cla()
            

if __name__ == "__main__":
    main(sys.argv[1:])