# import matplotlib
# matplotlib.use("agg")

# import sys
# import os
# import numpy as np
# from skimage.io import imread
# from sklearn.cluster import KMeans
# import pylab

# URL = '.\\data\\parrots.jpg'

# image = imread(URL)
# pylab.imshow(image)\

import matplotlib

import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

URL = '.\\data\\parrots.jpg'

image = imread(URL)
plt.imshow(image)