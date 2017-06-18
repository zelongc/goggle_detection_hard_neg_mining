'''
filename: visualization.py

Created on May 10, 2017

@author: Zelong Cong , University of Melbourne
'''

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import glob,os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s","--dir",required=True,help="The path to source directory containing the images")
args=vars(ap.parse_args())                          # input path must be absolute path

def visualization(path):
    image = color.rgb2gray(data.load(path))         #Load the image and convert it to grayscale.

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),     # compute the HOG features
                        cells_per_block=(2, 2), visualise=True,transform_sqrt=True,block_norm='L2-Hys')

    print("number of features: ",len(fd))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Cell size 4 x 4')
    ax1.set_adjustable('box-forced')
    plt.show()

    #Iterate all images in the folder
def test_folder(foldername):
    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        print('Now doing the visualization of ',filename)
        visualization(filename)

if __name__=="__main__":
    test_folder(args["dir"])


