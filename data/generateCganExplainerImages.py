import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from cgan_explainer import Cgan_Explainer
from mnistCnn import MnistCnn

(X_train, y_train), (_,_) = tf.keras.datasets.mnist.load_data()
img_data = X_train.copy()
img_data = img_data.reshape(X_train.shape[0], 28, 28, 1)
img_data = (img_data - 127.5) / 127.5
print(img_data.shape)

discriminator = load_model('mnist_cgan/mnist_cgan_100_epochs/cgan_discriminator.h5')
#discriminator.summary()
cnn = MnistCnn()
cnn.loadWeights("mnist_cnn/MnistCnnTEST")

explainer = Cgan_Explainer()

numImages = 50
indices = np.random.randint(0, img_data.shape[0], numImages) #random selection of images
time_start = time.time()
fig, axs = plt.subplots(10, 10, sharey=False, tight_layout=True, figsize=(16,16), facecolor='white')
for i in range(numImages):
    img = img_data[indices[i]]
    img_ = tf.cast(np.expand_dims(img, 0), tf.float32)
    x = tf.Variable(img_)
    cat = tf.constant(np.expand_dims(y_train[indices[i]],0))
    oldY = np.argmax(cnn.predict(x).numpy().squeeze())
    explainer.explain(x, classifier = cnn.model, discriminator = discriminator, maxIter = 600, maxChange = 0.099, minAlpha = 1000000, discWeight = 1)
    newY = np.argmax(cnn.predict(x).numpy().squeeze())
    title0 = str(oldY)
    title1 = str(newY)
    currentRow = math.floor(i/5)
    currentCol = 2*(i%5)
    axs[currentRow, currentCol].imshow(img, cmap = 'gray')
    axs[currentRow, currentCol].set_title(title0)
    axs[currentRow, currentCol+1].imshow(x[0], cmap = 'gray')
    axs[currentRow, currentCol+1].set_title(title1)
plt.savefig('cgan_explainer_images.png')