import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math
sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_wgan')
sys.path.append('../natural_adv')
import mnistCnn
import wgan
import recursiveSearch

def main(argv):
    outPath = './natAdvDiffImages'
    targetIndex = 1
    seed = 42
    opts, args = getopt.getopt(argv, "ho:c:s:", ['help', 'ofile=', 'class=', 'seed='])
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
        
    makeImage(outPath, targetIndex, seed)

def makeImage(outPath, targetIndex, seed = 42):
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    # pick random images
    np.random.seed(seed)
    index = np.random.randint(0, test_data.shape[0])
    image = test_data[index]
    label = y_test[index]
    # load gan
    gan = wgan.WGan()
    genPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_generator.h5"
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    invPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_inverter.h5"
    gan.load_gan(genPath, discPath, invPath)
    # load cnn
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    # generate adversary
    recSearch = recursiveSearch.RecursiveSearch()
    adversary = recSearch.search(gan.generator, gan.inverter, cnn.model, np.expand_dims(image, 0), label, y_t = targetIndex)
    x_adversary = adversary['x_adv']
    # prepare plot
    fig, ax = plt.subplots(1, 3, sharey = True, figsize=(12,6))
    # get predictions
    x = tf.Variable(tf.cast(np.expand_dims(image, 0), tf.float32))
    x_hat = tf.Variable(tf.cast(np.expand_dims(x_adversary, 0), tf.float32))
    pred_orig = tf.squeeze(cnn.model(x)).numpy()
    pred_orig_i = np.argmax(pred_orig)
    pred_orig_p = pred_orig[pred_orig_i]
    pred_new = tf.squeeze(cnn.model(x_hat)).numpy()
    pred_new_i = np.argmax(pred_new)
    pred_new_p = pred_new[pred_new_i]
    # get difference image
    diff_img = x_adversary - image
    print("diff in ", np.min(diff_img), ", ", np.max(diff_img))
    title0 = "original\n" + "pred: " + str(pred_orig_i) + "\nconf: " + str(round(pred_orig_p, 3))
    title1 = "new\n" + "pred: " + str(pred_new_i) + "\nconf: " + str(round(pred_new_p, 3))
    title2 = "$ \\Delta x $"
    ax[0].imshow(image, cmap = 'gray', interpolation = 'none')
    ax[0].set_title(title0)
    ax[1].imshow(x_adversary, cmap = 'gray', interpolation = 'none')
    ax[1].set_title(title1)
    ax[2].imshow(diff_img, cmap = 'seismic', interpolation = 'none', vmin = -1.0, vmax = 1.0)
    ax[2].set_title(title2)
    plt.savefig(outPath + "/" + "nat_adv_targetClass_{:04d}_".format(targetIndex) + "_seed_{:04d}.png".format(seed))
    
def printHelp(arg):
    msg = """
        usage: {} -o output_path -c targetClass -s seed""".format(arg)
    print(msg)
    
if __name__ == "__main__":
    main(sys.argv[1:])