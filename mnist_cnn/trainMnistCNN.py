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
    opts, args = getopt.getopt(argv, "hi:o:e:", ["ifile=", "ofile=", "epochs="])
    for opt, arg in opts:
        if opt == "-h":
            printHelp()
            sys.exit()
        elif(opt in ("-i", "--ifile")):
            inputPath = arg
        elif(opt in ("-o", "--ofile")):
            outputPath = arg
        elif(opt in ("-e", "--epochs")):
            epochs = arg
            if(int(epochs) < 1):
                print("epochs must be greater than 0.")
    
    batchSize = 64
    trainCnn(inputPath, outputPath, int(epochs), batchSize)
    
def trainCnn(inputPath, outputPath, epochs, batchSize):
    cnn = MnistCnn()
    if(inputPath != ""):
        cnn.loadWeights(inputPath)
    loader = MnistDatasetLoader()
    dataset = loader.loadDataset(batchSize)
    cnn.train(dataset, epochs, batchSize)
    cnn.saveModel(outputPath, overwrite = True)
    
if(__name__ == "__main__"):
    main(sys.argv[1:])