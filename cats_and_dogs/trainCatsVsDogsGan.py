import sys
import getopt

from catsVsDogsGan import CatsVsDogsGan
from datasetLoader import DatasetLoader

def main(argv):
    inputPath = ""
    outputPath = "cats_vs_dogs_GAN"
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
    trainGan(inputPath, outputPath, int(epochs))

def trainGan(inputPath, outputPath, epochs):
    gan = CatsVsDogsGan()
    if(inputPath != ""):
        genPath = inputPath + "/cats_vs_dogs_gen"
        discPath = inputPath + "/cats_vs_dogs_disc"
        gan.loadWeights(genPath, discPath)
    loader = DatasetLoader()
    catImgPath = "../../datasets/PetImages/Cat/"
    dogImgPath = "../../datasets/PetImages/Dog/"
    cat = 'cat'
    dog = 'dog'
    loader.addPathsFromDirectory(catImgPath, cat)
    loader.addPathsFromDirectory(dogImgPath, dog)
    loader.makeDataframe()
    batchSize = 32
    testDS, valDS, trainDS = loader.makeDatasets(128, batchSize, testSize = 500, validationSize = 500)
    gan.train(trainDS, epochs, batchSize, exampleFrequency = 5)
    genPath = outputPath + "/cats_vs_dogs_gen"
    discPath = outputPath + "/cats_vs_dogs_disc"
    gan.saveModel(genPath, discPath, overwrite = True)

def printHelp():
    helpMessage = """
    python trainCatsVsDogsGan.py -i existingGan -o targetDir -e epochs
    -i / --ifile:  train the existing gan at the given filepath
    -o / --ofile:  specify where to save the trained model
    -e / --epochs: specify the number of epochs to train for
    """
    print(helpMessage)


if(__name__ == "__main__"):
    main(sys.argv[1:])