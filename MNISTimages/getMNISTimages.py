import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train_classes), (x_test, y_test_classes) = mnist.load_data()
print(y_train_classes.shape)

fig = plt.figure()
img_counter = 0
for img in x_train:
    img_counter = img_counter+1
    print(img_counter)
    plt.imshow(img, cmap = 'gray')
    plt.savefig('MNISTimages/img_{:04d}.png'.format(img_counter))
    plt.cla
    if(img_counter >= 60):
        break