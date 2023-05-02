import mnist
from scipy import linalg
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

train_images = mnist.train_images()  # (60000, 28, 28)
train_labels = mnist.train_labels()  # (60000,)

test_images = mnist.test_images()  # (10000, 28, 28)
test_labels = mnist.test_labels()  # (10000,)

# reshape 3-d arrays into 2-d arrays
train_x = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))  # (60000, 784)
test_x = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))  # (10000, 784)






