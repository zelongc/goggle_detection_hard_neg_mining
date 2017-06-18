'''
filename: hard_neg_mining.py
Created on May 10, 2017
@author: Zelong Cong , University of Melbourne
 
This application is used to train a almost false-positive free model. it keeps updating models
from an initial one. It firstly applies the latest model to the negative samples, any features cause
positive predicitons are added to the negative features. then train the new classifier...

After training a new model, the application gives a confusion matrix.

'''

import numpy as np
from skimage.transform import pyramid_gaussian
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from skimage import color
import os
import glob
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--neg_image",required=True,help="The path to directory contains neg samples")
ap.add_argument("-p","--pos_feature",required=True,help="The path to pos features")
ap.add_argument("-n","--neg_feature",required=True,help="The path to neg features")
ap.add_argument("-r","--round",default="1",required=False,help="The starting round of hard negative mining")
ap.add_argument("-m","--model",required=True,help="The fist model used in hard negatie training")

args=vars(ap.parse_args())                          # input path must be absolute path

N=0   # initial a counter, used to count the nubmer of features in a single round.
number_of_round=args["round"]

def sliding_window(image, window_size, step_size):
    '''
    
    :param image: the input image
    :param window_size: the size of each window
    :param step_size: the size that each step takes, its a tuple (x,y), x and y stands for x,y axises repectively.
    :return: a window of image
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


def detect(filename,feature_folder,model_for_detection):

    im = cv2.imread(filename)          # read image
    im = imutils.resize(im, width=min(600, im.shape[1]))  # resize the image to width=600 pixels.

    min_wdw_sz = (130, 52)
    step_size = (10, 10)
    downscale = 1.15       # set parameter for sliding windows.

    clf = joblib.load(model_for_detection)    # load the classifier
    scale = 0                                 # initial a counter to count the level of image-pyramid

    for im_scaled in pyramid_gaussian(im, downscale=downscale):

        # the image-pyramid stops when the size of scaled iamge is smaller than the min window
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            im_window = color.rgb2gray(im_window)   # convert the window to grayscale
            im_window.astype(np.uint8)

            # Compute the HOG features and reshape the feature.
            fd = hist=hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
            fd = fd.reshape(1, -1)

            pred = clf.predict(fd)   # Classify the featrue

            ## When the pred = 1, the classifier classify the window as the object. which is a miss hit.
            ## then this feature is added to the feature directory, waiting next round training.
            if pred == 1:
                if clf.decision_function(fd) > 0.5:
                    global N
                    global number_of_round
                    name = 'round_'+str(number_of_round)+'_'+str(N) + '.feat'
                    joblib.dump(hist, feature_folder+ '/' + name)
                    N=N+1
        scale += 1

def walk_folder(foldername,feature_folder,model_for_detection):
    filenames = glob.iglob(os.path.join(foldername, '*'))

    ## apply detection to each negative image
    for filename in filenames:
        print(filename)
        detect(filename,feature_folder,model_for_detection)

# retrieve the folder then return a list of current files (images in this programme)
def find_names(object):
    for folder,dirnames,file_names in object:
        return file_names


def train_svm(pos_folder,neg_folder,model_name):

    feature_list = []
    label_list = []                                 # Initial two list to store features and labels.

    object_pos = os.walk(pos_folder)                # walk through the directory
    file_names_pos = find_names(object_pos)         # find a list of features' file names

    ## In this part, pos/neg features are loaded and labeled
    for each in file_names_pos:
        if 'feat' in each:
            feature = joblib.load(os.path.join(pos_folder, each))
            feature_list.append(feature)
            label_list.append(1)

    object_neg = os.walk(neg_folder)
    file_names_neg = find_names(object_neg)
    for each in file_names_neg:
        if 'feat' in each:
            feature = joblib.load(os.path.join(neg_folder, each))
            feature_list.append(feature)
            label_list.append(0)

    feature_list = np.array(feature_list, dtype=np.float)        # Convert the list to numpy object to improve efficiency
    feature_list = feature_list.reshape(feature_list.shape[:2])  # Reshap the Numpy object

    ## finishing generating the features, now print and check the size of lable/features list
    print("label size:",len(label_list))
    print("features size: ",np.array(feature_list).shape)

    ## train SVM
    print('finished generating the features, now start training the SVM.')
    classifier=LinearSVC(C=0.01)

    classifier.fit(feature_list,label_list)
    joblib.dump(classifier, model_name)

    result=classifier.predict(feature_list)

    global number_of_round
    print('finish training model for round %d' % number_of_round)
    print('accuracy: ',accuracy_score(label_list,result))
    print("this is the confusion metrics:")
    print(confusion_matrix(label_list,result))


def begin_boost():
    global number_of_round
    ## read arguments from command line
    neg_pic_folder=args["neg_image"]
    pos_feature_folder = args["pos_feature"]
    neg_feature_folder = args["neg_feature"]
    model_for_detection=args["model"]   # the first model used to apply detection.

    while 1:
        # generate and store all the False Negative
        if not os.path.isdir(neg_feature_folder):
            os.mkdir(neg_feature_folder)

        ## Begin hard_neg_training with latest model,
        walk_folder(neg_pic_folder,neg_feature_folder,model_for_detection)

        # Give new model's a name by the number of round.
        # For example, "train_hard_round_3.model" is the model trained for the 3rd round.
        model_name='train_hard_round_'+str(number_of_round)+'.model'

        train_svm(pos_feature_folder,neg_feature_folder,model_name)

        # preparing to begin next round
        number_of_round=number_of_round+1
        global N
        N=0
        print("*******************************")
        print('Finished round %d, now start round %d ' %(number_of_round,number_of_round-1))
        model_for_detection = model_name  # use the latest model for next round


if __name__=="__main__":
    begin_boost()




