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
    outPath = './singleDiffImages'
    targetIndex = 1
    seed = 42
    distance = 'l1'
    l_H = 1.0
    l_d = 0.0
    l_D = 0.0
    l_U = 0.0
    opts, args = getopt.getopt(argv, "ho:c:d:H:l:D:U:s:", ['help', 'ofile=', 'class=', 'distance=', 'lambdaH=', 'lambdad=', 'lambdaD=', 'lambdaU=', 'seed='])
    for opt, arg in opts:
        if(opt in ('-h', '--help')):
            printHelp(sys.argv[0])
            sys.exit()
        elif(opt in ('-o', '--outPath')):
            outPath = arg
        elif(opt in ('-c', '--class')):
            try:
                targetIndex = int(arg)
                if(targetIndex > 9 or targetIndex < 0):
                    print(arg, " must be an integer between 0 and 9")
                    sys.exit()
            except ValueError:
                print(arg, " must be an integer between 0 and 9")
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
        elif(opt in ('-s', '--seed')):
            try:
                seed = int(arg)
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
        
    makeImage(outPath, targetIndex, distance, l_H, l_d, l_D, l_U, seed)

def makeImage(outPath, targetIndex, distance = 'l1', l_H = 1.0, l_d = 0.0, l_D = 1.0, l_U = 1.0, seed = 42):
    print(distance)
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    # pick random images
    np.random.seed(seed)
    index = np.random.randint(0, test_data.shape[0])
    image = np.expand_dims(test_data[index],0)
    # load models
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    discriminator = wganDiscriminator.WganDiscriminator()
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    discriminator.load_discriminator(discPath)
    # init explainer
    explainer = explainerV2.Explainer(cnn.model, discriminator.discriminator)
    # prepare plot
    fig, ax = plt.subplots(1, 3, sharey = True, figsize=(12,6))
    # init variables
    x = tf.Variable(tf.cast(image, tf.float32))
    x_hat = tf.Variable(tf.identity(x))
    # get original prediction
    pred_orig = tf.squeeze(cnn.model(x)).numpy()
    pred_orig_i = np.argmax(pred_orig)
    pred_orig_p = pred_orig[pred_orig_i]
    # optimize
    steps = explainer.explain(x_hat, targetIndex, distance = distance, l_H = l_H, l_d = l_d, l_D = l_D, l_U = l_U)
    # get new prediction
    pred_new = tf.squeeze(cnn.model(x_hat)).numpy()
    pred_new_i = np.argmax(pred_new)
    pred_new_p = pred_new[pred_new_i]
    # get difference image
    diff_img = tf.Variable(tf.identity(x_hat))
    diff_img.assign(x_hat - x)
    print("diff in ", tf.reduce_min(diff_img).numpy(), ", ", tf.reduce_max(diff_img).numpy())
    title0 = "original\n" + "pred: " + str(pred_orig_i) + "\nconf: " + str(round(pred_orig_p, 3))
    title1 = "new\n" + "pred: " + str(pred_new_i) + "\nconf: " + str(round(pred_new_p, 3))
    title2 = "$ \\Delta x $"
    ax[0].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
    ax[0].set_title(title0)
    ax[1].imshow(x_hat[0,:,:,0], cmap = 'gray', interpolation = 'none')
    ax[1].set_title(title1)
    ax[2].imshow(diff_img[0,:,:,0], cmap = 'seismic', interpolation = 'none', vmin = -1.0, vmax = 1.0)
    ax[2].set_title(title2)
    plt.savefig(outPath + "/" + "targetClass_{:04d}_".format(targetIndex) + distance + "_seed_{:04d}.png".format(seed))
    
def printHelp(arg):
    msg = """
        usage: {} -o output_path -c targetClass -d distance -H lambda_H -l lambda_d -D lambda_D -U lambda_U -s seed""".format(arg)
    print(msg)
    
if __name__ == "__main__":
    main(sys.argv[1:])