'''
filename: gather_feature.py

Created on May 10, 2017

@author: Zelong Cong , University of Melbourne

This application is used to extract HOG features from pos/neg samples.
'''

from sklearn.externals import joblib
from skimage.feature import hog
import cv2
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p","--pos_image",required=True,help="The path to pos images directory")
ap.add_argument("-n","--neg_image",required=True,help="The path to neg images directory")
ap.add_argument("-f","--pos_feature",required=True,help="The path to pos feature directory")
ap.add_argument("-g","--neg_feature",required=True,help="The path to neg feature directory")

# The directory 'pos_feature' and 'neg_feature' are where the pos and neg features located, they must
# be different directories

args=vars(ap.parse_args())     # input path must be absolute path

SZ=20
bin_n = 16                     # Number of bins to caculate the histogram
N=0

def compute_hog(img_path,type):
    img=cv2.imread(img_path,0)     # read the image in graysacle (0 stands fro grayscale)

    hist=hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualise=False, transform_sqrt=True,
        feature_vector=True)       # comput the HOG feature os the image.

    if type=='pos':
        feature_folder=args["pos_feature"]     # save positive image's features in positive feature directory
    else:
        feature_folder=args["neg_feature"]     # save neg image's features in negative feature directory

    if not os.path.isdir(feature_folder):
        os.mkdir(feature_folder)
    global N
    name=str(N)+'.feat'
    joblib.dump(hist,feature_folder+'/'+name)  # save the image's features to the corresponding directories.
    N=N+1

# retrieve the folder then return a list of current files (images in this programme)
def find_names(object):
    for folder,dirnames,file_names in object:
        return file_names

# this funciton is used to iterate all the image from the directory.
def compute_feature(folder_path,type):
    object = os.walk(folder_path)
    pic_names = find_names(object)
    for each in pic_names:
        if each is not None and "DS_Store" not in each :
            img_path=os.path.join(folder_path,each)
            compute_hog(img_path,type)

if __name__ == '__main__':

    compute_feature(args["pos_image"],'pos')
    compute_feature(args["neg_image"],'neg')
