# create and train a gan for the cats and dogs dataset

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
import time
import matplotlib.pyplot as plt
import os

class CatsVsDogsGan:

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(8*8*256, use_bias = False, input_shape = (100, )))
        model.add(Reshape((8,8,256)))
        assert model.output_shape == (None,8,8,256)
        model.add(Conv2DTranspose(256, (4, 4), strides = (1, 1), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(32, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(3, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, activation = 'tanh'))
        return model
    
    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, (4, 4), strides = (2, 2), padding = 'same', use_bias = False, input_shape = [128, 128, 3]))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(48, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(96, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(192, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(384, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.fill(real_output.shape, 0.9), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, images, batchSize):
        noiseDim = 100
        noise = tf.random.normal([batchSize, noiseDim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self.generator(noise, training = True)
            fake_output = self.discriminator(fake_imgs, training = True)
            real_output = self.discriminator(images, training = True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))
    
    def train(self, dataset, epochs, batchSize, verbose = 0, exampleFrequency = 0):
        for epoch in range(epochs):
            start = time.time()
            img_batch_counter = 0
            for image_batch in dataset:
                self.train_step(image_batch[0], batchSize)
                img_batch_counter = img_batch_counter+1
            if(verbose == 1):
                print("time for epoch ", epoch+1, " is ", time.time()-start)
            if((exampleFrequency > 0) and (epoch+1 % exampleFrequency == 0)):
                self.saveExample(epoch)
        print("Training done.")

    def loadWeights(self, genPath, discPath):
        if(os.path.exists(genPath) and os.path.exists(discPath)):
            self.generator = tf.keras.models.load_model(genPath)
            self.discriminator = tf.keras.models.load_model(discPath)
        else:
            print("Can't find one or both of the specified paths.")
    
    def saveModel(self, genPath, discPath, overwrite = False):
        if((os.path.exists(genPath) or os.path.exists(discPath)) and overwrite == False):
            msg = """At least one of the specified paths already exists.
                    Use overwrite = True if you want to overwrite the existing directory."""
            print(msg)
        else:
            self.generator.save(genPath)
            self.discriminator.save(discPath)
    
    def saveExample(self, epoch):
        example = self.generator(tf.random.normal([1,100]))
        fig = plt.figure(figsize = (8,4))
        plt.imshow(example[0][:][:][:])
        plt.savefig('img_at_epoch_{:04d}.png'.format(epoch))
        
    def __init__(self):
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        lr = 1e-4
        beta1 = 0.5
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta1, beta_2 = 0.999)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = beta1, beta_2 = 0.999)