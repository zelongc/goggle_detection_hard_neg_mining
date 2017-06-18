'''
filename: image_resize.py

Created on May 10, 2017

@author: Zelong Cong , University of Melbourne

This application resizes all the images in one directory and store the images in one directory.
user can choose the flip mode to save a flipping version of each image.
'''
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t","--target_path",required=True,help="The path to store the resized images")
ap.add_argument("-s","--image_path",required=True,help="The path to directory contains images to be resized")
ap.add_argument("-f","--flip",default='No',required=False,help="whether we need to flip the images")

args=vars(ap.parse_args())

target_path= args["target_path"]   # Load target path.
flip=args['flip']

if os.path.isdir(target_path):
    pass
else:
    os.mkdir(target_path)     # If the target path does not exist in OS, create one
num=0

def resize(img_path,name):
    img=cv2.imread(img_path)

    # Resize the image to a certain size.
    img_resized=cv2.resize(img,(60,77))

    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    global num
    name=str(num)+'_'+name
    cv2.imwrite(os.path.join(target_path,name),img_resized)

    if flip == "yes" :                       # If it is flip mode, flip the image and store it
        cv2.imwrite(os.path.join(target_path,'reversed_'+name),cv2.flip(img_resized,1))
        print('finish resizing file %s' % name)
        num=num+1


# Walk through the directory and resize the images
def find_names1(object):
    for folder,dirnames,file_names in object:
        for each in file_names:
            if  "DS_Store" not in each:
                resize(os.path.join(folder,each),each)


# This method is used when the directory contains the images has sub directories.
def find_names(object):
    for folder,dirnames,file_names in object:
        for each in dirnames:
            object2= os.walk(os.path.join(folder,each))
            for a,b,c in object2:
                for each2 in c:
                    print(os.path.join(a,each2))
                    resize(os.path.join(a,each2),each2)

if __name__=="__main__":

    file_path=args["image_path"]
    object=os.walk(file_path)
    pic_names=find_names1(object)
