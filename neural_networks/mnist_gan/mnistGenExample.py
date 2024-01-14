import tensorflow as tf
from mnistGan import MnistGan
import matplotlib.pyplot as plt

gan = MnistGan()
genPath = './NewMnistGan200Epochs/mnist_gen'
discPath = './NewMnistGan200Epochs/mnist_disc'
gan.loadWeights(genPath, discPath)

example = gan.generator(tf.random.normal([16,100]), training = False)
#print(example)
fig = plt.figure(figsize=(4,4))
for i in range(example.shape[0]):
    plt.subplot(4,4,i+1)
    plt.imshow(example[i,:,:,0], cmap = 'gray')
    plt.axis('off')
plt.savefig('NewMnistGenExample200Epochs.png')