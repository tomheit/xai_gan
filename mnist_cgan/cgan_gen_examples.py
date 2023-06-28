import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import sys
import os
import getopt

from mnist_cgan import Mnist_Cgan

def main(argv):
    inputPath = ""
    outputPath = ""
    opts, args = getopt.getopt(argv, "i:o:", ["ifile=", "ofile="])
    for opt, arg in opts:
        if(opt in ("-i", "--ifile")):
            inputPath = arg
        elif(opt in ("-o", "--ofile")):
            outputPath = arg
            
    latent_points, _ = latent_vector(100,100)
    labels = np.asarray([x for _ in range(10) for x in range(10)])
    model = load_model(inputPath + '/cgan_generator.h5')
    gen_images = model.predict([latent_points,labels])
    gen_images = (gen_images+1)/2.0
    fig, axs = plt.subplots(10, 10, sharey=False, tight_layout=True, figsize=(16,16), facecolor='white')
    k=0
    for i in range(0,10):
        for j in range(0,10):
            axs[i,j].matshow(gen_images[k], cmap='gray')
            axs[0,j].set(title=labels[k])
            axs[i,j].axis('off')
            k=k+1
    plt.savefig(outputPath + "/cgan_gen_example.png")
    
def latent_vector(latent_dim, n, n_cats=10):
    latent_input = np.random.randn(latent_dim * n)
    latent_input = latent_input.reshape(n, latent_dim)
    cat_labels = np.random.randint(0, n_cats, n)
    return [latent_input, cat_labels]
    
if(__name__ == "__main__"):
    main(sys.argv[1:])