import sys
import getopt
import os
import tensorflow as tf

from mnistCnn import MnistCnn
from mnistDatasetLoader import MnistDatasetLoader

def main(argv):
    inputPath = ""
    outputPath = "mnist_cnn"
    epochs = 100
    largerModel = False
    opts, args = getopt.getopt(argv, "hi:o:e:l", ["ifile=", "ofile=", "epochs="])
    for opt, arg in opts:
        if opt == "-h":
            printHelp()
            sys.exit()
        elif(opt == "-l"):
            largerModel = True
        elif(opt in ("-i", "--ifile")):
            inputPath = arg
        elif(opt in ("-o", "--ofile")):
            outputPath = arg
        elif(opt in ("-e", "--epochs")):
            epochs = arg
            if(int(epochs) < 1):
                print("epochs must be greater than 0.")
    
    batchSize = 128
    trainCnn(inputPath, outputPath, int(epochs), batchSize, largerModel)
    
def trainCnn(inputPath, outputPath, epochs, batchSize, largerModel = False):
    cnn = MnistCnn(largerModel)
    if(inputPath != ""):
        cnn.loadWeights(inputPath)
    loader = MnistDatasetLoader()
    dataset = loader.loadDataset(batchSize)
    cnn.train(dataset, epochs, batchSize)
    cnn.saveModel(outputPath, overwrite = True)
    
def printHelp():
    helpMsg = """
    python trainMnistCNN.py -i existingCNN -o targetDir -e epochs
    -i / --ifile:  train the existing model at the given filepath
    -o / --ofile:  specify where to save the trained model
    -e / --epochs: specify the number of epochs to train for
    """
    print(helpMsg)
    
if(__name__ == "__main__"):
    main(sys.argv[1:])