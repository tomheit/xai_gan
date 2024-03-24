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
import shap

def main(argv):
    outPath = './shap'
    seed = 42
    distance = 'l1_mad'
    l_H = 0.085
    l_d = 0.000005
    l_D = 0.01
    l_U = 1.0
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
        
    makeImages(outPath, distance, l_H, l_d, l_D, l_U)
    
def makeImages(outPath, distance, l_H, l_d, l_D, l_U):
    # get mnist data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255
    # pick random images
    np.random.seed(42)
    index = np.random.randint(0,test_data.shape[0])
    test_image = test_data[index]
    # load models
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/m3_150Epochs')
    discriminator = wganDiscriminator.WganDiscriminator()
    discPath = "../neural_networks/mnist_wgan/wGan_1000_epochs/wgan_discriminator.h5"
    discriminator.load_discriminator(discPath)
    # init explainer
    explainer = explainerV2.Explainer(cnn.model, discriminator.discriminator)
    # shap preparation
    indices = np.random.randint(0,test_data.shape[0],100)
    background = test_data[indices]
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
    explainerShap = shap.DeepExplainer(cnn.model, background)
    shap_values = explainerShap.shap_values(np.expand_dims(test_image,0))
    print(shap_values.shape)
    swapped = np.swapaxes(shap_values, 0, -1)
    shap_numpy = [np.swapaxes(np.swapaxes(np.swapaxes(s, 1, 2), 0, 1), 0, -1) for s in swapped]
    shap.image_plot(shap_numpy, -np.expand_dims(test_image, 0))
    #plt.savefig(outPath + "/shapComp_seed_{:04d}.png".format(42))
    #sys.exit()
    #
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
        if(j == orig_pred):
                ax[0,j+1].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
                ax[0,j+1].axis('off')
                ax[0,j+1].set_title('original')
                ax[1,j+1].imshow(x[0,:,:,0], cmap='gray', interpolation='none')
                ax[1,j+1].axis('off')
                tempImg = ax[2,j+1].imshow(shap_values[0,:,:,0,j], cmap = 'seismic')
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
        tempImg = ax[2,j+1].imshow(shap_values[0,:,:,0,j], cmap = 'seismic')
        ax[2,j+1].axis('off')
        #tempImg.set_norm(norm)
    plt.subplots_adjust(wspace = 0.0, hspace = 0.0)
    plt.savefig(outPath + "/shapComp_seed_{:04d}.png".format(42))
    
def printHelp():
    msg = """
    usage: python shapExplanation.py -o outpath -d distance -H l_H -l l_d -D l_D -U l_U -n numImgs
    -o: output directory for image
    -d: string describing distance, one of l1, l2, l1_mad
    -H: lambda_H weight crossentropy loss in target function
    -l: lambda_d weight of distance in target function
    -D: lambda_D weight of discriminator in target function
    -U: lambda_U weight of U(x) in target function
    use a weight of 0.0 to disable the term in the target function"""
    print(msg)
    
if __name__ == "__main__":
    main(sys.argv[1:])