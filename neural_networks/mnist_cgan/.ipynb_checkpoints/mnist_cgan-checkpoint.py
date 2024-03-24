import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import numpy as np

class Mnist_Cgan:
    def __init__(self):
        latent_dim = 100
        self.generator = self.makeGenerator(latent_dim)
        self.discriminator = self.makeDiscriminator()
        self.gan = self.makeGan(self.generator, self.discriminator)
        
    def makeGenerator(self, latent_dim, in_shape = (7,7,1), n_cats = 10):
        #label input
        in_label = Input(shape = (1,), name = 'Generator-Label-Input-Layer') #input
        labels = Embedding(n_cats, 50, name = 'Generator-Label-Embedding-Layer')(in_label)

        n_nodes = in_shape[0]*in_shape[1]
        labels = Dense(n_nodes, name = 'Generator-Label-Dense-Layer')(labels)
        labels = Reshape((in_shape[0], in_shape[1], 1), name = 'Generator-Label-Reshape-Layer')(labels)

        #latent space input
        in_latent = Input(shape = latent_dim, name = 'Generator-Latent-Input-Layer')

        n_nodes = 7*7*128
        g = Dense(n_nodes, name = 'Generator-Foundation-Layer')(in_latent)
        g = ReLU(name = 'Generator-Foundation-Layer-Activation-1')(g)
        g = Reshape((in_shape[0], in_shape[1], 128), name = 'Generator-Foundation-Layer-Reshape-1')(g)

        #concatenate both inputs
        concat = Concatenate(name = 'Generator-Combine-Layer')([g, labels])

        #Hidden Layer 1
        g = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = 'same', name = 'Generator-Hidden-Layer-1')(concat)
        g = ReLU(name = 'Generator-Hidden-Layer-Activation-1')(g)

        #Hidden Layer 2
        g = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = 'same', name = 'Generator-Hidden-Layer-2')(g)
        g = ReLU(name = 'Generator-Hidden-Layer-Activation-2')(g)

        #Output Layer (only one filter because the image is greyscale, not rgb)
        output_layer = Conv2D(filters = 1, kernel_size = (7,7), activation = 'tanh', padding = 'same', name = 'Generator-Output-Layer')(g)

        #define and return model
        model = Model([in_latent, in_label], output_layer, name = 'Generator')
        return model
    
    def makeDiscriminator(self, in_shape = (28,28,1), n_cats = 10):
        #Label Input
        in_label = Input(shape = (1,), name = 'Discriminator-Label-Input-Layer')
        labels = Embedding(n_cats, 50, name = 'Discriminator-Label-Embedding-Layer')(in_label)

        n_nodes = in_shape[0] * in_shape[1]
        labels = Dense(n_nodes, name = 'Discriminator-Label-Dense-Layer')(labels)
        labels = Reshape((in_shape[0], in_shape[1], 1), name = 'Discriminator-Label-Reshape-Layer')(labels)

        #Image Input
        in_image = Input(shape = in_shape, name = 'Discriminator-Image-Input-Layer')

        #concatenate inputs
        concat = Concatenate(name = 'Discriminator-Combine-Layer')([in_image, labels])

        #Hidden Layer 1
        h = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'Discriminator-Hidden-Layer-1')(concat)
        h = LeakyReLU(alpha = 0.2, name = 'Discriminator-Hidden-Layer-Activation-1')(h)

        #Hidden Layer 2
        h = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'Discriminator-Hidden-Layer-2')(h)
        h = LeakyReLU(alpha = 0.2, name = 'Discriminator-Hidden-Layer-Activation-2')(h)
        h = MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'valid', name = 'Discriminator-Max-Pool-Layer-2')(h)

        #Flatten and Output Layer
        h = Flatten(name = 'Discriminator-Flatten-Layer')(h)
        h = Dropout(0.2, name = 'Discriminator-Flatten-Layer-Dropout')(h)
        output_layer = Dense(1, activation = 'sigmoid', name = 'Discriminator-Output-Layer')(h)

        #def model
        model = Model([in_image, in_label], output_layer, name = 'Discriminator')
        model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5), metrics = ['accuracy'])
        return model
    
    def makeGan(self, generator, discriminator):
        discriminator.trainable = False
        gen_latent, gen_label = generator.input
        gen_output = generator.output
        gan_output = discriminator([gen_output, gen_label])
        model = Model([gen_latent, gen_label], gan_output, name = 'cDCGAN')
        model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5))
        return model
    
    def load_gan(self, genPath, disPath):
        self.generator = load_model(genPath)
        self.discriminator = load_model(disPath)
        self.gan = self.makeGan(self.generator, self.discriminator)
    
    def load_generator(self, genPath):
        self.generator = load_model(genPath)
        self.gan = self.makeGan(self.generator, self.discriminator)
        
    def load_discriminator(self, disPath):
        self.discriminator = load_model(disPath)
        self.gan = self.makeGan(self.generator, self.discriminator)