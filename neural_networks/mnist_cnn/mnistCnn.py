import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, LeakyReLU, ReLU, Reshape, Softmax
import os

class MnistCnn:
    def __init__(self, largerModel = False):
        self.imgHeight = 28
        self.imgWidth = 28
        self.numClasses = 10
        if(largerModel):
            self.model = self.makeM3()
        else:
            self.model = self.makeMedModel()
    
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
    
    def makeLargerModel(self): #https://www.kaggle.com/code/seyyedrezamoslemi/digit-recognizer
        cnn_model = tf.keras.models.Sequential()
        cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5), input_shape = (self.imgHeight, self.imgWidth, 1)))
        cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Flatten())
        cnn_model.add(Dense(32, activation = 'relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(self.numClasses, activation = 'softmax'))
        return cnn_model
    
    def makeM3(self):
        shape = (self.imgHeight, self.imgWidth, 1)
        cnn_model = tf.keras.models.Sequential()
        cnn_model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = shape, use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 48, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 80, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 96, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 112, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 144, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 160, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Conv2D(filters = 176, kernel_size = (3,3), use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(ReLU())
        cnn_model.add(Flatten())
        cnn_model.add(Dense(self.numClasses, use_bias = False))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Softmax())
        return cnn_model
    
    def makeMedModel(self):
        cnn_model = tf.keras.models.Sequential()
        cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5), input_shape = (self.imgHeight, self.imgWidth, 1)))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
        cnn_model.add(MaxPooling2D())
        cnn_model.add(Dropout(0.3))
        cnn_model.add(BatchNormalization())
        cnn_model.add(Flatten())
        cnn_model.add(Dense(128, activation = 'relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(self.numClasses, activation = 'softmax'))
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
