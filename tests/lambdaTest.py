# **** compare different weights lambda_H for the loss function only ****
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
    # start and stop values
    start = 0.001
    stop = 1.0
    outPath = "./"
    opts, args = getopt.getopt(argv, "a:b:o:h")
    for opt, arg in opts:
        if(opt == '-h'):
            printHelp()
            sys.exit()
        elif(opt == '-o'):
            outPath = arg
        elif(opt == '-a'):
            try:
                start = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        elif(opt == '-b'):
            try:
                stop = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
                sys.exit()
        else:
            print("Unknown parameter: {0}".format(opt))
    if(os.path.exists(outPath)):
        print("Directory already exists. Existing files may be replaced.")
    else:
        print("Directory does not exist. Creating directory...")
        os.makedirs(outPath)
        
    makeImage(start, stop, outPath)

def makeImage(start, stop, outPath = './'):
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    test_img = test_data[0]
    # load models
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    discriminator = wganDiscriminator.WganDiscriminator()
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    discriminator.load_discriminator(discPath)
    # init explainer
    explainer = explainerV2.Explainer(cnn.model, discriminator.discriminator)
    # step size
    t = abs(stop - start)/5.0
    # prepare plot
    fig, ax = plt.subplots(1,6, sharey = True, figsize=(10,5))
    for i in range(6):
        l_t = start + (i * t)
        x = tf.Variable(tf.cast(np.expand_dims(test_img,0), tf.float32))
        steps = explainer.explain(x, 1, distance = 'l1', l_H = l_t, l_d = 0.0, l_D = 0.0, l_U = 0.0)
        new_pred = tf.squeeze(cnn.model(x)).numpy()
        pred_class = np.argmax(new_pred)
        conf = new_pred[1]
        title = "pred: " + str(pred_class) + "\n conf: " + str(round(conf, 3)) + "\n steps: " + str(steps) + "\n l_H: " + str(round(l_t, 3))
        ax[i].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
        ax[i].set_title(title)
        ax[i].axis('off')
    plt.subplots_adjust(bottom = 0.01, wspace = 0.0, hspace = 0.4)
    plt.savefig(outPath + "/img_{:04f}.png".format(start))
            
def printHelp():
    msg = "usage: python lambdaTest.py -a start_value -b stop_value -o output_directory"
    print(msg)

if __name__ == "__main__":
    main(sys.argv[1:])