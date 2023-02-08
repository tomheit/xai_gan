# create a dataset from images

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import glob
import cv2

class DatasetLoader:
    
    def __init__(self):
        self.imagePaths = []
        self.data = {}
        self.dataframe = pd.DataFrame()
        self.imageSize = 0
    
    def addPathsFromDirectory(self, path, label):
        paths = glob.glob(path + '*.jpg')
        self.imagePaths = self.imagePaths + paths
        # add paths and label to dictionary
        for i in paths:
            self.data[i] = label
    
    def makeDataframe(self):
        self.dataframe = pd.DataFrame(self.imagePaths)
        self.dataframe.rename({0:'path'}, axis = 1, inplace = True)
        self.dataframe['classname'] = self.dataframe['path'].map(self.data)
        classnames = self.dataframe['classname'].unique()
        K = classnames.size
        name2class = dict(zip(classnames, range(K)))
        self.dataframe['class'] = self.dataframe['classname'].map(name2class)

    def makeDatasets(self, imgSize, batchSize, testSize = 0, validationSize = 0):
        self.imageSize = imgSize
        numImages = self.dataframe.shape[0]
        allIdxs = np.array(range(numImages))
        np.random.shuffle(allIdxs)
        testDF = self.dataframe.iloc[allIdxs[0:testSize]]
        valDF = self.dataframe.iloc[allIdxs[testSize:testSize+validationSize]]
        trainDF = self.dataframe.iloc[allIdxs[testSize+validationSize:]]
        testDS = self.makeDataset(testDF, batchSize)
        valDS = self.makeDataset(valDF, batchSize)
        trainDS = self.makeDataset(trainDF, batchSize)
        return testDS, valDS, trainDS
    
    def pathToArray(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels = 3)
        img = tf.cast(img, dtype = tf.float32)/255
        img = tf.image.resize(img, [self.imageSize, self.imageSize])
        img = tf.image.random_flip_left_right(img)
        #num_rot = np.random.randint(0,4)
        img = tf.image.random_brightness(img, max_delta = 0.1)
        img = tf.image.random_hue(img, max_delta = 0.02)
        img = tf.image.random_saturation(img, lower = 0.95, upper = 1.05)
        label = tf.one_hot(label, depth = 2)
        return img, label
    
    def makeDataset(self, df, batchSize):
        dsPath = tf.data.Dataset.from_tensor_slices((df['path'],df['class']))
        ds = dsPath.map(self.pathToArray)
        ds = ds.batch(batchSize)
        return ds
