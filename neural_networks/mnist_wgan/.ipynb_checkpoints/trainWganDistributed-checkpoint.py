import os
import sys, getopt
import tensorflow as tf
import matplotlib.pyplot as plt
import wgan

def main(argv):
    
    opts, args = getopt.getopt(argv, "e:d:c:")
    for opt, arg in opts:
        if(opt == '-e'):
            epochs = int(arg)
            saveDirectory = './wGan_' + str(epochs) + '_epochs'
            exampleDir = saveDirectory + '/training_examples'
        elif(opt == '-d'):
            saveDirectory = arg
        elif(opt == '-c'):
            loadDirectory = arg
            loadFromDir = True
        else:
            print("Unknown parameter: {0}".format(opt))
            
    epochs = 100 #100000 epochs used in https://github.com/zhengliz/natural-adversary/tree/master
    batchSize = 32
    bufferSize = 60000
    sampleFrequency = 100 #generate sample images periodically during training
    saveDirectory = './wGan_' + str(epochs) + '_epochs'
    loadDirectory = './wGan'
    loadFromDir = False
    exampleDir = saveDirectory + '/training_examples'
    
    # https://www.tensorflow.org/guide/distributed_training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    # load mnist dataset
    (X_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    img_data = X_train.copy()
    img_data = img_data.reshape(X_train.shape[0], 28, 28, 1)
    img_data = img_data / 255 #pixel values in [0,1]
    train_dataset = tf.data.Dataset.from_tensor_slices((img_data)).shuffle(bufferSize).batch(batchSize)
    distributed_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    
    gan = wgan.WGan(distributedTraining)
    
    if(loadFromDir):
        gan.load_gan(loadDirectory + '/wgan_generator.h5', loadDirectory + '/wgan_discriminator.h5', loadDirectory + '/wgan_inverter.h5')
    if(not os.path.exists(saveDirectory)):
        os.makedirs(saveDirectory)
    if(not os.path.exists(exampleDir)):
        os.makedirs(exampleDir)
    gan.train_loop(train_dataset, epochs, batchSize, exampleDir = exampleDir, CRITIC_ITERS = 5, exampleFrequency = 100, verbose = True)
    gan.saveModels(saveDirectory)
    
if __name__ == "__main__":
    main(sys.argv[1:])