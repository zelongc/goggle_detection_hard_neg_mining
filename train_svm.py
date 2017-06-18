'''
filename: train_svm.py

Created on May 10, 2017

@author: Zelong Cong , University of Melbourne
'''
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p","--pos_feature",required=True,help="The path to pos features directory")
ap.add_argument("-n","--neg_feature",required=True,help="The path to neg features directory")
ap.add_argument("-s","--model_name",required=True,help="name of the model")

# The directory 'pos_feature' and 'neg_feature' are where the pos and neg features located, they must
# be different directories

args=vars(ap.parse_args())

# retrieve the folder then return a list of current files (images in this programme)
def find_names(object):
    for folder,dirnames,file_names in object:
        return file_names

# produce the confusion matrix and ROC graph using test samples.
def produce_report(classifier,Test_X,Test_Y):

    result = classifier.predict(Test_X)        # make prediciton of Test samples,

    print('accuracy: ', accuracy_score(Test_Y, result))  # produce accuracy of model based on test samples
    print("matrix: ")
    print(confusion_matrix(Test_Y, result))
    print('test size is', len(Test_Y))

    ## Draw ROC of the model.
    prob = classifier._predict_proba_lr(Test_X)
    _prob = []
    for each in prob:
        _prob.append(each[1])

    fpr, tpr, thresholds = roc_curve(Test_Y, _prob)
    roc_auc = auc(fpr, tpr)
    print('False positive rate is', fpr)
    print('True positive rate is ', tpr)
    print("thresholds are ", thresholds)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def train_svm(pos_folder,neg_folder):

    feature_list=[]
    label_list=[]         # Initial two list to store features and labels.

    object_pos=os.walk(pos_folder)     # walk through the directory
    file_names_pos=find_names(object_pos)      # find a list of features' file names

    # In this part, pos/neg features are loaded and labeled
    for each in file_names_pos:
        if 'feat' in each:
            feature=joblib.load(os.path.join(pos_folder,each))
            feature_list.append(feature)
            label_list.append(1)

    object_neg = os.walk(neg_folder)
    file_names_neg = find_names(object_neg)

    for each in file_names_neg:
        if 'feat' in each:
            feature = joblib.load(os.path.join(neg_folder, each))
            feature_list.append(feature)
            label_list.append(0)

    feature_list=np.array(feature_list,dtype=np.float)          # Convert the list to numpy object to improve efficiency
    feature_list=feature_list.reshape(feature_list.shape[:2])   # Reshap the Numpy object

    ## finishing generating the features, now print and check the size of lable/features list
    print("label size:",len(label_list))
    print("features size: ",np.array(feature_list).shape)


    print('finished generating the features, now start training the SVM.')
    classifier=LinearSVC(C=0.01)  # small C means large Margin---overfit data

    joblib.dump(classifier, args["model_name"])    # save model.

    ## Split data, 'Train' are used to train model and the 'Test' are used to test model
    ## test_size is the ratio of testing samples to training samples %65 samples used to train classifier
    ## and 35% samples are used to test in this scenario.
    Train_X,Test_X,Train_Y,Test_Y=train_test_split(feature_list,label_list,test_size=0.35,random_state=0)

    ## train the classifier
    classifier.fit(Train_X,Train_Y)

    # Produce report by testing data
    produce_report(classifier,Test_X,Test_Y)


if __name__ == '__main__':

    train_svm(args["pos_feature"],args["neg_feature"])