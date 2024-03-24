import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_wgan')
sys.path.append('../explainer')
sys.path.append('../natural_adv')

import explainer
import mnistCnn
import WGan
import recursiveSearch

def main(argv):
    cnnPath = '../neural_networks/mnist_cnn/largerCnn30Epochs'
    genPath = '../neural_networks/mnist_wgan/wgan_1000_epochs/wgan_generator.h5'
    discPath = '../neural_networks/mnist_wgan/wgan_1000_epochs/wgan_discriminator.h5'
    invPath = '../neural_networks/mnist_wgan/wgan_1000_epochs/wgan_inverter.h5'
    numberOfExamples = 10
    (X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_data = X_train.copy()
    img_data = img_data.reshape(X_train.shape[0], 28, 28, 1)
    img_data = img_data / 255 #pixel values in [0,1]
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    y_test_one_hot = np.eye(10)[y_test]
    # get random samples
    np.random.seed(seed = 2)
    indices = np.random.randint(0, test_data.shape[0], numberOfExamples)
    #load models
    cnn = mnistCnn.MnistCnn()
    gan = wgan.WGan()
    cnn.loadWeights(cnnPath)
    gan.load_gan(genPath, discPath, invPath)
    
    real_images = test_data[indices]
    numberOfExplainers = 10
    fig, ax = plt.subplots(numberOfExamples, 11)
    for img in real_images:
        

if __name__ == "__main__":
    main(sys.argv[1:])