import sys
import getopt
import os
import tensorflow as tf
import numpy as np

from mnistCnn import MnistCnn

def main(argv):
    inputPath = ""
    epochs = 50
    opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    for opt, arg in opts:
        if opt == "-h":
            printHelp()
            sys.exit()
        elif(opt in ("-i", "--ifile")):
            inputPath = arg
    
    testCnn(inputPath)
    
def testCnn(inputPath):
    # load data
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    test_data = x_test.copy()
    test_data = test_data.reshape(x_test.shape[0], 28, 28, 1)
    test_data = test_data / 255 # normalize
    y_test_one_hot = np.eye(10)[y_test] # one-hot encode labels
    cnn = MnistCnn()
    cnn.loadWeights(inputPath)
    test_results = cnn.model.evaluate(test_data, y_test_one_hot)
    print("test loss, test acc: ", test_results)
    
def printHelp():
    helpMsg = """
    python testMnistCNN.py -i existingCNN
    -i / --ifile:  test the existing model at the given filepath
    """
    print(helpMsg)
    
if(__name__ == "__main__"):
    main(sys.argv[1:])