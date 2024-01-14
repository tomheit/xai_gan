import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import sys
import os
import getopt

from mnist_cgan import Mnist_Cgan

def main(argv):
    inputPath = ""
    outputPath = "mnist_cgan"
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
                
    data, categories = load_dataset()
    gan = Mnist_Cgan()
    
    if(inputPath != ""):
        genPath = inputPath + '/cgan_generator.h5'
        disPath = inputPath + '/cgan_discriminator.h5'
    
    trainGan(gan.generator, gan.discriminator, gan.gan, data, categories, latent_dim = 100, n_epochs = int(epochs))
    gan.generator.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate = 0.0002, beta_1 = 0.5))
    gan.generator.save(outputPath + '/cgan_generator.h5')
    gan.discriminator.save(outputPath + '/cgan_discriminator.h5')

def load_dataset():
    (X_train, y_train), (_,_) = keras.datasets.mnist.load_data()
    img_data = X_train.copy()
    img_data = img_data.reshape(X_train.shape[0], 28, 28, 1)
    img_data = (img_data - 127.5) / 127.5
    return img_data, y_train

def real_samples(dataset, categories, n):
    indx = np.random.randint(0, dataset.shape[0], n)
    X, cat_labels = dataset[indx], categories[indx]
    y = np.ones((n,1))
    return [X, cat_labels], y

def latent_vector(latent_dim, n, n_cats=10):
    latent_input = np.random.randn(latent_dim * n)
    latent_input = latent_input.reshape(n, latent_dim)
    cat_labels = np.random.randint(0, n_cats, n)
    return [latent_input, cat_labels]

def fake_samples(generator, latent_dim, n):
    latent_output, cat_labels = latent_vector(latent_dim, n)
    X = generator.predict([latent_output, cat_labels])
    y = np.zeros((n,1))
    return [X, cat_labels], y

def trainGan(g_model, d_model, gan_model, dataset, categories, latent_dim, n_epochs = 10, n_batch = 128, n_eval = 200):
    batch_per_epoch = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            #train disc
            [x_real, cat_labels_real], y_real = real_samples(dataset, categories, half_batch)
            discriminator_loss1, _ = d_model.train_on_batch([x_real, cat_labels_real], y_real)
            [x_fake, cat_labels_fake], y_fake = fake_samples(g_model, latent_dim, half_batch)
            discriminator_loss2, _ = d_model.train_on_batch([x_fake, cat_labels_fake], y_fake)
            #train gen
            [latent_input, cat_labels] = latent_vector(latent_dim, n_batch)
            y_gan = np.ones((n_batch,1))
            generator_loss = gan_model.train_on_batch([latent_input, cat_labels], y_gan)
        print(i, " epochs done")

if(__name__ == "__main__"):
    main(sys.argv[1:])