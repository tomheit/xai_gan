import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_wgan')
sys.path.append('../explainer')
import explainerV2
import mnistCnn
import wganDiscriminator
from skimage.color import gray2rgb, rgb2gray, label2rgb
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

def main(argv):
    outPath = './lime'
    seed = 42
    distance = 'l1_mad'
    l_H = 0.085
    l_d = 0.000005
    l_D = 0.01
    l_U = 1.0
    opts, args = getopt.getopt(argv, "ho:d:H:l:D:U:n:s:", ['help', 'ofile=', 'distance=', 'lambdaH=', 'lambdad=', 'lambdaD=', 'lambdaU=', 'numImgs=', 'seed='])
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
        
    makeImages(outPath, distance, l_H, l_d, l_D, l_U, seed)
    
def makeImages(outPath, distance, l_H, l_d, l_D, l_U, seed):
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    # pick random images
    np.random.seed(seed)
    index = np.random.randint(0,test_data.shape[0])
    test_image = test_data[index]
    # load models
    global cnn
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    discriminator = wganDiscriminator.WganDiscriminator()
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    discriminator.load_discriminator(discPath)
    # init explainer
    explainer = explainerV2.Explainer(cnn.model, discriminator.discriminator)
    # lime preparation
    explainerLIME = lime_image.LimeImageExplainer(verbose = False)
    segmenter = SegmentationAlgorithm('felzenszwalb', scale=3, sigma=0.8, min_size=4)
    explanation = explainerLIME.explain_instance(gray2rgb(test_image[:,:,0]), classifier_fn = rgbModel, top_labels = 10, hide_color=0, num_samples=100, segmentation_fn = segmenter)
    red = [(1/255.0)*181, (1/255.0)*22, (1/255.0)*33]
    blue = [(1/255.0)*0, (1/255.0)*100, (1/255.0)*173]
    colors3 = [red,[0.0,0.0,0.0],blue]
    x = tf.Variable(tf.cast(np.expand_dims(test_image,0), tf.float32))
    orig_pred = np.argmax(tf.squeeze(cnn.model(x)).numpy())
    fig, ax = plt.subplots(3, 11, sharey=True, figsize=(20,5))
    norm = colors.Normalize(vmin = -10, vmax = 10)
    ax[0,0].imshow(test_image, cmap='gray',interpolation='none')
    ax[0,0].set_title("original")
    ax[0,0].axis('off')
    ax[2,0].imshow(test_image,cmap='gray',interpolation='none')
    ax[2,0].axis('off')
    ax[1,0].imshow(test_image,cmap='gray',interpolation='none')
    ax[1,0].axis('off')
    for j in range(10):
        tempImg, mask = explanation.get_image_and_mask(j, positive_only=False, num_features=10, hide_rest=False, min_weight = 0.001)
        maskedTempImg = label2rgb(3-mask, test_image[:,:,0], colors=colors3, bg_label=0, saturation = 0)
        if(j == orig_pred):
                ax[0,j+1].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
                ax[0,j+1].axis('off')
                ax[0,j+1].set_title('original')
                ax[1,j+1].imshow(x[0,:,:,0], cmap='gray', interpolation='none')
                ax[1,j+1].axis('off')
                tempImg = ax[2,j+1].imshow(maskedTempImg, interpolation = 'none')
                ax[2,j+1].axis('off')
                #tempImg.set_norm(norm)
                continue
        x_hat = tf.Variable(tf.identity(x))
        steps = explainer.explain(x_hat, j, distance = distance, l_H = l_H, l_d = l_d, l_D = l_D, l_U = l_U)
        diff = x_hat-x
        pred = np.argmax(tf.squeeze(cnn.model(x_hat)).numpy())
        title = "f(x): " + str(pred)
        ax[0,j+1].imshow(x_hat[0,:,:,0], cmap = 'gray', interpolation = 'none')
        ax[0,j+1].axis('off')
        ax[0,j+1].set_title(title)
        ax[1,j+1].imshow(diff[0,:,:,0], cmap='seismic',interpolation='none', vmin=-1,vmax=1)
        ax[1,j+1].axis('off')
        tempImg = ax[2,j+1].imshow(maskedTempImg, interpolation = 'none')
        ax[2,j+1].axis('off')
        #tempImg.set_norm(norm)
    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.savefig(outPath + "/limeComp_seed_{:04d}.png".format(seed))
    
# takes an rgb image and converts it to grayscale
# then calls the model
def rgbModel(x):
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    grayImages = rgb2gray(x)
    if(len(grayImages.shape) == 3):
        return cnn.model(np.expand_dims(grayImages, -1))
    else:
        return cnn.model(np.expand_dims(np.expand_dims(grayImages, -1), 0))
    
def printHelp():
    msg = "lol"
    print(msg)
    
if __name__ == "__main__":
    main(sys.argv[1:])