# create and train a cnn model to work on the cats and dogs dataset

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
import os

class CatsVsDogsCnn:
    self.kernel_size = (3, 3)
    self.pool_size   = (2, 2)
    self.first_filters  = 32
    self.second_filters = 64
    self.third_filters  = 128
    self.dropout_conv  = 0.3
    self.dropout_dense = 0.3
    self.isCompiled = False

    def make_cnn_model(self):
        cnn_model = tf.keras.models.Sequential()

        cnn_model.add( BatchNormalization(input_shape = (img_size, img_size, 3)))
        cnn_model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
        cnn_model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
        cnn_model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
        cnn_model.add( MaxPooling2D (pool_size = pool_size)) 
        cnn_model.add( Dropout (dropout_conv))

        cnn_model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
        cnn_model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
        cnn_model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
        cnn_model.add( MaxPooling2D (pool_size = pool_size))
        cnn_model.add( Dropout (dropout_conv))

        cnn_model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
        cnn_model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
        cnn_model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
        cnn_model.add( MaxPooling2D (pool_size = pool_size))
        cnn_model.add( Dropout (dropout_conv))

        cnn_model.add( Flatten())
        cnn_model.add( Dense (256, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)))
        cnn_model.add( Dropout (dropout_dense))
        cnn_model.add( Dense (2, activation = "softmax"))
        return cnn_model

    def __init__(self):
        model = make_cnn_model()
    
    def saveModel(self, path, overwrite = False):
        if(os.path.exists(path)):
            if(overwrite == False):
                print(path, " already exists. Please specify overwrite = True if you want to overwrite the existing directory.")
            else:
                model.save(path)
                print("saved model to ", path)
        else:
            model.save(path)
            print("saved model to ", path)
    
    def loadWeights(self, path):
        if(os.path.exists(path)):
            model = tf.keras.models.load_model('cats_vs_dogs_cnn_model')
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss = 'binary_crossentropy', metrics = ['accuracy'])
            isCompiled = True
        else:
            print("Can't find ", path)
    
    def train(
        self,
        dataset, val_data, epochs = 50,
        steps_per_epoch = 600,
        checkpointFilename = 'cats_vs_dogs_cnn.h5'):
        if(isCompiled == False):
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss = 'binary_crossentropy', metrics = ['accuracy'])
            isCompiled = True
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
        modelfname, monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 0)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss', mode = 'min', factor = 0.75, patience = 4, verbose = 0)

        history = model.fit(
            dataset, epochs = epochs,
            steps_per_epoch = steps_per_epoch,
            validation_data = val_data,
            verbose = 0,
            callbacks = [checkpoint, reduce_lr])
        
        print("Training done.")
        