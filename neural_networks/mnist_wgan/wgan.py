import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape, ReLU, Input, Cropping2D
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.initializers
import time
import os
import math
import matplotlib.pyplot as plt
from matplotlib import colors

# wgan based on https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py
# inverter based on https://github.com/zhengliz/natural-adversary/tree/master

class WGan:
    def __init__(self, distributedTraining = False):
        self.DIM = 64
        self.CRITIC_ITERS = 5
        self.LAMBDA = 10
        self.OUTPUT_DIM = 28*28
        self.LATENT_DIM = 128
        self.LAMBDA_INV = 0.1
        self.generator = self.makeGeneratorModel()
        self.discriminator = self.makeDiscriminatorModel()
        self.inverter = self.makeInverterModel()
        self.generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.inverterOptimizer = tf.keras.optimizers.Adam(1e-4)
        #if(distributedTraining):
        #    with mirrored_strategy.scope():
        #        self.generator = self.makeGeneratorModel()
        #        self.discriminator = self.makeDiscriminatorModel()
        #        self.inverter = self.makeInverterModel()
        #        self.generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        #        self.discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        #        self.inverterOptimizer = tf.keras.optimizers.Adam(1e-4)
        
    def makeGeneratorModel(self, latent_dim = 128):
        DIM = self.DIM
        OUTPUT_DIM = self.OUTPUT_DIM
        input_layer = Input(shape = 128)
        g = Dense(4*4*4*DIM, activation = 'relu')(input_layer)
        g = Reshape([4, 4, 4*DIM])(g)
        # he normal initializer used by gulrajani
        g = Conv2DTranspose(filters = 2*DIM, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(g)
        g = ReLU()(g)
        g = Cropping2D(cropping = ((0,1), (0,1)), data_format = 'channels_last')(g)
        g = Conv2DTranspose(filters = DIM, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(g)
        g = ReLU()(g)
        output_layer = Conv2DTranspose(filters = 1, kernel_size = 5, strides = (2,2), padding = 'SAME', activation = 'sigmoid', kernel_initializer = 'he_normal')(g)
        #output_layer = Reshape([OUTPUT_DIM])(g)
        model = Model(input_layer, output_layer, name = 'Generator')
        return model
        
    def makeDiscriminatorModel(self):
        DIM = self.DIM
        input_layer = Input(shape = (28, 28, 1))
        d = Conv2D(filters = DIM, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(input_layer)
        d = LeakyReLU()(d)
        d = Conv2D(filters = 2*DIM, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(d)
        d = LeakyReLU()(d)
        d = Conv2D(filters = 4*DIM, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(d)
        d = LeakyReLU()(d)
        d = Reshape([4*4*4*DIM])(d)
        d = Dense(1)(d)
        output_layer = Reshape([])(d)
        model = Model(input_layer, output_layer, name = 'Discriminator')
        return model
    
    def makeInverterModel(self, latent_dim = 128):
        z_dim = latent_dim
        input_layer = Input(shape = (28, 28, 1))
        i = Conv2D(filters = latent_dim, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(input_layer)
        i = LeakyReLU()(i)
        i = Conv2D(filters = 2*latent_dim, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(i)
        i = LeakyReLU()(i)
        i = Conv2D(filters = 4*latent_dim, kernel_size = 5, strides = (2,2), padding = 'SAME', kernel_initializer = 'he_normal')(i)
        i = LeakyReLU()(i)
        i = Reshape([latent_dim * 64])(i)
        i = Dense(latent_dim * 8)(i)
        output_layer = Dense(z_dim)(i)
        model = Model(input_layer, output_layer, name = 'Inverter')
        return model
    
    def latent_vector(self, batchSize):
        vector = tf.random.normal(shape = [batchSize, self.LATENT_DIM])
        return vector
        
    @tf.function
    def train_step_disc(self, imageBatch, batchSize):
        real_data = tf.cast(imageBatch, tf.float32) #real images
        z = self.latent_vector(batchSize) #latent input for generator
        fake_data = self.generator(z) #generated images
        alpha = tf.random.uniform(shape = [batchSize,1,1,1], minval = 0, maxval = 1) #alpha val
        interpolates = (alpha * real_data) + (fake_data - (alpha * fake_data))
        with tf.GradientTape() as disc_tape:
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolates)
                D_interpolates = self.discriminator(interpolates)
            grad_interpolates = gp_tape.gradient(D_interpolates, interpolates)
            disc_loss = tf.reduce_mean(self.discriminator(fake_data)) - tf.reduce_mean(self.discriminator(real_data))
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_interpolates)))
            disc_loss += self.LAMBDA*tf.reduce_mean((slopes - 1)**2)
        grad_disc_loss = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminatorOptimizer.apply_gradients(zip(grad_disc_loss, self.discriminator.trainable_variables))
        
    @tf.function
    def train_step_gen(self, imageBatch, batchSize):
        z = self.latent_vector(batchSize) #latent input for generator
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(z) #generated images
            gen_loss = -tf.reduce_mean(self.discriminator(fake_data))
        grad_gen_loss = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generatorOptimizer.apply_gradients(zip(grad_gen_loss, self.generator.trainable_variables))
        
    @tf.function
    def train_step_inv(self, imageBatch, batchSize):
        l = self.LAMBDA_INV
        z = self.latent_vector(batchSize) #z from latent space
        x_gen = self.generator(z) #generated images
        x = tf.cast(imageBatch, tf.float32)
        with tf.GradientTape() as inv_tape:
            x_rec = self.generator(self.inverter(x)) #recreated images from inverted z
            z_rec = self.inverter(x_gen) #latent values from inverted images
            inv_loss = tf.reduce_mean(tf.square(x - x_rec)) + l*tf.reduce_mean(tf.square(z - z_rec))
        grad_inv_loss = inv_tape.gradient(inv_loss, self.inverter.trainable_variables)
        self.inverterOptimizer.apply_gradients(zip(grad_inv_loss, self.inverter.trainable_variables))
            
    # unable to generate examples if @tf.function
    def train_loop(self, dataset, epochs, batchSize, exampleDir, CRITIC_ITERS = 5, exampleFrequency = 100, verbose = False):
        for epoch in range(epochs):
            startTime = time.time()
            critic_counter = 0
            for imageBatch in dataset:
                critic_counter += 1;
                if(critic_counter == CRITIC_ITERS):
                    critic_counter = 0
                    self.train_step_gen(imageBatch, batchSize)
                self.train_step_disc(imageBatch, batchSize)
                self.train_step_inv(imageBatch, batchSize)
            if(exampleFrequency > 0 and (epoch % exampleFrequency == 0)):
                self.saveExample(epoch, exampleDir)
            if(verbose == True):
                deltaTime = time.time()-startTime
                print("epoch {:04d} done after: ".format(epoch), deltaTime)
                remainingEpochs = epochs - (epoch+1)
                remainingTime = (remainingEpochs * deltaTime)/(60*60)
                print("estimated time remaining: ", remainingTime, " h")
                
    def load_gan(self, genPath, discPath, invPath):
        self.generator = load_model(genPath)
        self.discriminator = load_model(discPath)
        self.inverter = load_model(invPath)
        
    def load_generator(self, genPath):
        self.generator = load_model(genPath)
        
    def load_discriminator(self, discPath):
        self.discriminator = load_model(discPath)
        
    def load_inverter(self, invPath):
        self.inverter = load_model(invPath)
        
    def save_generator(self, path):
        self.generator.save(path + '/wgan_generator.h5')
        
    def save_discriminator(self, path):
        self.discriminator.save(path + '/wgan_discriminator.h5')
        
    def save_inverter(self, path):
        self.inverter.save(path + '/wgan_inverter.h5')
        
    def saveModels(self, directory):
        self.save_generator(directory)
        self.save_discriminator(directory)
        self.save_inverter(directory)
        
    def saveExample(self, epoch, exampleDir):
        norm = colors.Normalize(vmin = 0, vmax = 1)
        latentVec = self.latent_vector(16)
        generatedImages = self.generator(latentVec, training = False)
        fig, ax = plt.subplots(4, 4)
        for i in range(16):
            img = ax[math.floor(i/4), i%4].imshow(generatedImages[i,:,:,0], cmap = 'gray', interpolation = 'none')
            img.set_norm(norm)
        plt.savefig(exampleDir + '/img_at_epoch_{:04d}.png'.format(epoch))