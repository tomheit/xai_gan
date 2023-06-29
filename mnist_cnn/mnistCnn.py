import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, Reshape
import os

class MnistCnn:
    def __init__(self):
        self.imgHeight = 28
        self.imgWidth = 28
        self.numClasses = 10
        self.model = self.makeCnnModel()
    
    def makeCnnModel(self):
        cnn_model = tf.keras.models.Sequential()
        cnn_model.add(Conv2D(filters = 8, kernel_size = (3,3), input_shape = (self.imgHeight, self.imgWidth, 1)))
        cnn_model.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.25))
        cnn_model.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.25))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(self.numClasses, activation = 'softmax'))

        #cnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
        #                 loss = 'categorical_crossentropy',
        #                 metrics = ['accuracy'])
        #cnn_model.summary()
        return cnn_model
    
    def train(self, dataset, _epochs, batchSize):
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                         loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])
        history = self.model.fit(dataset, epochs = _epochs, batch_size = batchSize)
    
    def saveModel(self, path, overwrite = False):
        if(os.path.exists(path) and overwrite == False):
            print("The given directory already exists. Specify overwrite=True to overwrite it.")
        else:
            self.model.save(path)
    
    def loadWeights(self, path):
        if(os.path.exists(path)):
            self.model = tf.keras.models.load_model(path)
        else:
            print("Can't find the path: ", path)
    
    def predict(self, x):
        return self.model(x)
