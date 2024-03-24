# ***** This class contains only the discriminator from the wGan *****
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Reshape, Input
from tensorflow.keras.models import Model, load_model

class WganDiscriminator:
    def __init__(self):
        self.DIM = 64
        self.discriminator = self.makeDiscriminatorModel()
        
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
    
    def load_discriminator(self, discPath):
        self.discriminator = load_model(discPath)