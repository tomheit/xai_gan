import tensorflow as tf
from catsVsDogsGan import CatsVsDogsGan
import matplotlib.pyplot as plt

gan = CatsVsDogsGan()
genPath = './CatsVSDogsGanTEST/cats_vs_dogs_gen'
discPath = './CatsVSDogsGanTEST/cats_vs_dogs_disc'
gan.loadWeights(genPath, discPath)

example = gan.generator(tf.random.normal([1,100]))
#print(example)
fig = plt.figure(figsize = (8,4))
plt.imshow(example[0][:][:][:])
plt.savefig('CatsVsDogsGenExample.png')