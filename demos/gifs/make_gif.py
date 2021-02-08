import os
import imageio
import numpy as np    
from matplotlib import pyplot as plt

# Randomly creating 10 covariance matrices with an increasing variance
for i in range(10): 
    xs = np.random.normal(size=(100,6), scale=(i+1))
    xs_cov = np.cov(xs, rowvar=0)
    indx = '0'+str(i)
    if len(indx)==3:
        indx = indx[1:]
    plt.imshow(xs_cov)
    plt.savefig('./images/sim'+indx+'mycov.png')
    plt.close()

# Pulling in the images that were exported 
file_names = sorted((fn for fn in os.listdir('./images/') if fn.endswith('mycov.png')))
# Exporting the images to a gif
images = []
for filename in file_names:
    images.append(imageio.imread(filename))
imageio.mimsave('./images/sim_my_cov.gif', images)
