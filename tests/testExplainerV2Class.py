import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math
sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_wgan')
sys.path.append('../explainer')
import explainerV2
import mnistCnn
import wganDiscriminator

def main(argv):
    outPath = './tests'
    numberOfImages = 5
    distance = 'l1'
    l_H = 1.0
    l_d = 0.0
    l_D = 0.0
    l_U = 0.0
    opts, args = getopt.getopt(argv, "ho:d:H:l:D:U:n:", ['help', 'ofile=', 'distance=', 'lambdaH=', 'lambdad=', 'lambdaD=', 'lambdaU=', 'numImgs='])
    for opt, arg in opts:
        if(opt in ('-h', '--help')):
            printHelp()
            sys.exit()
        elif(opt in ('-o', '--outPath')):
            outPath = arg
        elif(opt in ('-d', '--distance')):
            distance = arg
        elif(opt in ('-H', '--lambdaH')):
            try:
                l_H = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        elif(opt in ('-l', '--lambdad')):
            try:
                l_d = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        elif(opt in ('-D', '--lambdaD')):
            try:
                l_D = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        elif(opt in ('-U', '--lambdaU')):
            try:
                l_U = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        elif(opt in ('-n', '--numImgs')):
            try:
                numberOfImages = int(arg)
            except ValueError:
                print(arg, " must be an integer!")
                sys.exit()
        else:
            print("Unknown parameter: {0}".format(opt))
    if(os.path.exists(outPath)):
        print("Directory already exists. Existing files may be replaced.")
    else:
        print("Directory does not exist. Creating directory...")
        os.makedirs(outPath)
        
    makeImages(numberOfImages, outPath, distance, l_H, l_d, l_D, l_U)
    
def makeImages(n, outPath, distance, l_H, l_d, l_D, l_U):
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    # pick random images
    np.random.seed(42)
    indices = np.random.randint(0, test_data.shape[0], n)
    # load models
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    discriminator = wganDiscriminator.WganDiscriminator()
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    discriminator.load_discriminator(discPath)
    # init explainer
    explainer = explainerV2.Explainer(cnn.model, discriminator.discriminator)
    fig, ax = plt.subplots(n, 11, sharey = True, figsize=(12,6))
    for j in range(n):
        x = tf.Variable(tf.cast(np.expand_dims(test_data[indices[j]],0), tf.float32))
        orig_pred = np.argmax(tf.squeeze(cnn.model(x)).numpy())
        ax[j,0].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
        ax[j,0].axis('off')
        ax[j,0].set_title('original')
        for i in range(10):
            if(i == orig_pred):
                ax[j,i+1].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
                ax[j,i+1].axis('off')
                ax[j,i+1].set_title('original')
                continue
            x_hat = tf.Variable(tf.identity(x))
            steps = explainer.explain(x_hat, i, distance = distance, l_H = l_H, l_d = l_d, l_D = l_D, l_U = l_U)
            pred = np.argmax(tf.squeeze(cnn.model(x_hat)).numpy())
            title = "f(x): " + str(pred)
            ax[j,i+1].imshow(x_hat[0,:,:,0], cmap = 'gray', interpolation = 'none')
            ax[j,i+1].axis('off')
            ax[j,i+1].set_title(title)
    plt.subplots_adjust(wspace = 0.0, hspace = 0.3)
    plt.savefig(outPath + "/" + distance + "_seed_{:04d}.png".format(42))
    
def printHelp():
    msg = "lol"
    print(msg)
    
if __name__ == "__main__":
    main(sys.argv[1:])