import tensorflow as tf
import numpy as np

class MnistDatasetLoader:
    def loadDataset(self, batchSize = 256):
        bufferSize = 60000
        # loading the mnist dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train_classes), (x_test, y_test_classes) = mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        # preparation
        num_classes = 10
        num_train = x_train.shape[0]
        num_test = x_train.shape[0]
        img_width = x_train.shape[1]
        img_height = x_train.shape[2]
        img_size_flat = img_width * img_height
        # one-hot encoding
        y_train = np.eye(num_classes, dtype = float)[y_train_classes]
        y_test = np.eye(num_classes, dtype = float)[y_test_classes]
        # normalize input to [0,1]
        x_train = x_train / 255
        x_test = x_test / 255
        # create tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(bufferSize).batch(batchSize)
        return train_dataset