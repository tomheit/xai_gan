import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
import time
import os
import matplotlib.pyplot as plt

class MnistGan:
    def __init__(self, simplifiedDiscriminator = False):
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits = False)
        self.generator = self.makeGeneratorModel()
        if(simplifiedDiscriminator):
            self.discriminator = self.makeSimplifiedDiscriminatorModel()
        else:
            self.discriminator = self.makeDiscriminatorModel()
        self.generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
        
    def makeGeneratorModel(self):
        model = tf.keras.Sequential()
        model.add(Dense(4*4*256, use_bias = False, input_shape = (100, )))
        model.add(Reshape((4,4,256)))
        assert model.output_shape == (None, 4, 4, 256)
        model.add(Conv2DTranspose(512, (4, 4), strides = (1, 1), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = 'same', output_padding = (1, 1), use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        #model.add(Dense(28*28, use_bias = False))
        #model.add(Reshape((28,28,1)))
        model.add(Conv2DTranspose(1, (1, 1), strides = (1, 1), padding = 'same', use_bias = False, activation = 'tanh'))
        #model.summary()
        assert model.output_shape == (None, 28, 28, 1)
        return model
    
    def CVDmakeGeneratorModel(self): #from catsvsdogs
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
        model.summary()
        return model
    
    def OLDmakeGeneratorModel(self):
        model = tf.keras.Sequential()
        model.add(Dense(7*7*256, use_bias = False, input_shape = (100, )))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)
        model.add(Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)
        return model
    
    def makeDiscriminatorModel(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', use_bias = False, input_shape = [28, 28, 1]))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(128, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(256, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(1, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))
        return model
    
    def makeSimplifiedDiscriminatorModel(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', use_bias = False, input_shape = [28, 28, 1]))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Conv2D(1, (4, 4), strides = (2, 2), padding = 'same', use_bias = False))
        model.add(Flatten())
        model.add(Dense(1, activation = 'sigmoid'))
        return model
    
    def OLDmakeDiscriminatorModel(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [28, 28, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1))
        return model
    
    def discriminator_loss(self, realOutput, fakeOutput):
        realLoss = self.crossEntropy(tf.fill(tf.shape(realOutput), 0.9), realOutput)
        fakeLoss = self.crossEntropy(tf.zeros_like(fakeOutput), fakeOutput)
        totalLoss = realLoss + fakeLoss
        return totalLoss

    def generator_loss(self, fakeOutput):
        return self.crossEntropy(tf.ones_like(fakeOutput), fakeOutput)
    
    @tf.function
    def trainStep(self, images, batchSize):
        noise = tf.random.normal([batchSize, 100])
        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.generator(noise, training = True)
            realOutput = self.discriminator(images, training = True)
            fakeOutput = self.discriminator(generatedImages, training = True)
            genLoss = self.generator_loss(fakeOutput)
            discLoss = self.discriminator_loss(realOutput, fakeOutput)
        
        gradientsOfGen = genTape.gradient(genLoss, self.generator.trainable_variables)
        gradientsOfDisc = discTape.gradient(discLoss, self.discriminator.trainable_variables)

        self.generatorOptimizer.apply_gradients(zip(gradientsOfGen, self.generator.trainable_variables))
        self.discriminatorOptimizer.apply_gradients(zip(gradientsOfDisc, self.discriminator.trainable_variables))
    
    def train(self, dataset, epochs, batchSize, exampleFrequency, verbose = False):
        for epoch in range(epochs):
            start = time.time()
            for imageBatch in dataset:
                self.trainStep(imageBatch[0], batchSize)
            if(exampleFrequency > 0 and (exampleFrequency % epochs+1 == 0)):
                self.saveExample(epoch)
            if(verbose == True):
                print("epoch {:04d} done after: ".format(epoch), time.time()-start)
        
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