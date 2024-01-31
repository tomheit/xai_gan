import sys, getopt
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.append('../neural_networks/mnist_cnn')
sys.path.append('../neural_networks/mnist_gan')
sys.path.append('../explainer')

import explainer
import mnistCnn
import mnistGan

def main(argv):
    validTargetFunctions = ['loss', 'loss_gan', 'loss_gan_u', 'loss_u',
                            'prob', 'prob_gan', 'prob_gan_u', 'prob_u',
                            'negative_prob', 'negative_prob_gan', 'negative_prob_gan_u', 'negative_prob_u']
    numberOfExamples = 100
    normConst = ""
    targetFunc = ""
    targetClass = "second"
    targetDir = ""
    diff_img = False
    epsilon = 0.1
    opts, args = getopt.getopt(argv, "n:c:t:s:d:o:e:h")
    for opt, arg in opts:
        if(opt == '-n'):
            numberOfExamples = int(arg)
        elif(opt == '-c'):
            if(arg == "2"):
                normConst = "euclidean"
            elif(arg == "1"):
                normConst = "one"
            elif(arg == "max"):
                normConst = "max"
            else:
                print("Invalid norm constraint. Select one of '2', '1', 'max'. Got: ", arg)
                sys.exit()
        elif(opt == '-t'):
            if(arg not in validTargetFunctions):
                print("Invalid target function. Select one of: ", validTargetFunctions, ". Got: ", arg)
                sys.exit()
            else:
                targetFunc = arg
        elif(opt == '-s'):
            if(arg == "second"):
                targetClass = arg
            elif(arg == "confusion"):
                targetClass = arg
            else:
                print("Invalid target class selection. Choose 'second' or 'confusion'. Got: ", arg)
                sys.exit()
        elif(opt == '-d'):
            targetDir = arg
        elif(opt == '-h'):
            print("usage: ", argv[0], " [-n NumberOfExamples] [-c NormConstraint] [-t TargetFunction] [-s TargetClassSelection] [-d Directory] [-e epsilon]")
            print("example: ", argv[0], " -n 100 -c 2 -t loss_gan -s second -d /.")
        elif(opt == '-o'):
            if(arg == 'diff'):
                diff_img = True
        elif(opt == '-e'):
            try:
                epsilon = float(arg)
            except ValueError:
                print(arg, " must be a floating point number!")
        else:
            print("Unknown parameter: {0}".format(opt))
    
    if(normConst == ""):
        print("Please specify which norm constraint to use.")
        sys.exit()
    if(targetFunc == ""):
        print("Please specify the target function.")
        sys.exit()
    if(targetClass == ""):
        print("Please specify the target class selection.")
        sys.exit()
    if(targetDir == ""):
        print("Please specify the target directory")
        sys.exit()
    else:
        if(os.path.exists(arg)):
            print("Directory already exists. Existing files may be replaced.")
        else:
            print("Directory does not exist. Creating directory...")
            os.makedirs(targetDir)
        
    cnn = mnistCnn.MnistCnn()
    cnn.loadWeights('../neural_networks/mnist_cnn/largerCnn30Epochs')
    #cnn.model.summary()

    #load gan
    genPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_gen'
    discPath = '../neural_networks/mnist_gan/NewMnistGan200Epochs/mnist_disc'
    gan = mnistGan.MnistGan()
    gan.loadWeights(genPath, discPath)

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
    
    # random images from the test data
    indices = np.random.randint(0, test_data.shape[0], numberOfExamples)
    # create explainer
    exp = explainer.Explainer(cnn.model, gan.discriminator)
    
    targetIndex = -1
    predictionsOfIndices = tf.argmax(cnn.model(test_data[indices]), axis = 1).numpy()
    print(predictionsOfIndices)
    if(targetClass == 'confusion'):
        predictionsProb = cnn.model(test_data)
        predictions = tf.argmax(predictionsProb, 1)
        confusionMatrix = tf.math.confusion_matrix(labels = y_test, predictions = predictions, num_classes = 10)
        #targetIndex = tf.math.top_k(confusionMatrix[0], k=2).indices.numpy()[1]
        #print("target index: ", targetIndex)
    
    n_images = 2
    if(diff_img == True):
        n_images = 3
    fig, ax = plt.subplots(1,n_images)
    for i in range(numberOfExamples):
        title = "img"+str(i)
        img = test_data[indices[i]]
        x = tf.Variable(tf.cast(np.expand_dims(img, 0), tf.float32))
        if(targetClass == 'confusion'):
            targetIndex = tf.math.top_k(confusionMatrix[predictionsOfIndices[i]], k=2).indices.numpy()[1]
            #print(targetIndex, type(targetIndex))
        nIter, originalPred, originalConf, newPred, newConf = exp.explain(x, targetIndex, epsilon = epsilon, normConstraint = normConst, targetFunction = targetFunc)
        title0 = "pred: " + str(originalPred) + ", p: " + str(originalConf)
        title1 = "pred: " + str(newPred) + ", p: " + str(newConf)
        ax[0].imshow(img, cmap = 'gray', interpolation = 'none')
        ax[0].set_title(title0)
        ax[1].imshow(x[0,:,:,0], cmap = 'gray', interpolation = 'none')
        ax[1].set_title(title1)
        if(diff_img == True):
            diff = ((x[0] - img)/2 + 0.5)
            ax[2].imshow(diff, cmap = 'seismic', interpolation = 'none')
            ax[2].set_title("diff")
        plt.savefig(targetDir + "/img_{:04d}.png".format(i))
        plt.cla()

if __name__ == "__main__":
    main(sys.argv[1:])