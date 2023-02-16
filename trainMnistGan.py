import sys
import getopt
import os
import tensorflow as tf

from mnistGan import MnistGan
from mnistDatasetLoader import MnistDatasetLoader

def main(argv):
    inputPath = ""
    outputPath = "mnist_gan"
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
    
    batchSize = 256
    trainGan(inputPath, outputPath, int(epochs), batchSize)
    
def trainGan(inputPath, outputPath, epochs, batchSize):
    gan = MnistGan()
    if(inputPath != ""):
        genPath = inputPath + "/mnist_gen"
        discPath = inputPath + "/mnist_disc"
        gan.loadWeights(genPath, discPath)
    loader = MnistDatasetLoader()
    dataset = loader.loadDataset(batchSize)
    gan.train(dataset, epochs, batchSize, exampleFrequency = 50)
    genPath = outputPath + "/mnist_gen"
    discPath = outputPath + "/mnist_disc"
    gan.saveModel(genPath, discPath, overwrite = True)
    
def printHelp():
    helpMessage = """
    python trainMnistGan.py -i existingGan -o targetDir -e epochs
    -i / --ifile:  train the existing gan at the given filepath
    -o / --ofile:  specify where to save the trained model
    -e / --epochs: specify the number of epochs to train for
    """
    print(helpMessage)

if(__name__ == "__main__"):
    main(sys.argv[1:])