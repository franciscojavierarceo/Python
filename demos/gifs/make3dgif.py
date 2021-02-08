import os
from images2gif import writeGif
from PIL import Image

file_names = [f for f in os.listdir('./') if f.endswith('.png')]
file_names.sort()
images = [Image.open(fn) for fn in file_names]
writeGif("3dplot_gif.GIF", images, duration=0.75)