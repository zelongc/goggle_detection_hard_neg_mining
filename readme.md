
# goggle detetion with HOG + SVM + Hard Neg Training

### Author: Zelong Cong   University of Melbonrne
### Date:   1st June, 2017

## Evironments:

1. OpenCV Module 3.2.0 (latest version)
   (This is a possible guide to install the module, though it not always works for some reasons...
     http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/)

2. Python 3.6.2

## A list of required packages:

1. scikit-learn
2. scikit-image
3. cv2 (python wrapper for OpenCV)
4. numpy
5. imutils
6. matplotlib

## What this archive contains

1.This archive contains 8 major applications created during the Project:

	'visualization.py'    ---- The application used to visualise the image's HOG features.

	'gather_feature.py'   ---- The application used to extract features.

	'train_svm.py'        ____ The application used to train a custom made SVM model.

	'hard_neg_mining.py'  ---- A very important application used to apply hard negative training process.

	'detection.py'    ---- The most siginificant application achieved our goal, to detect human and identify whether he is wearing a pair of goggle with OpenCV detector.

	'tool/Image_net_downloader.py'  ---- The application used to download images to local machine from ImageNet

	'tool/image_resize.py'     ---- The application used to resize the images including flipping operation.

	'tool/test_custom_detector.py'  ---- The application used to experiment the goggle detection.

2. The models for test locates in the directory 'models'

## Guide to test

IMPORTANT: better to use absolute path to run applicatioin.

1. To use the 'goggle_detection_opencv.py' 
```
Use command $ 'python3 goggle_detection_opencv.py -i path/to/an/image -f path/to/face/detector -g path/to/model/goggle'

for example in this directory, 

'python3 goggle_detection_opencv.py -i test_image/IMG_0518.JPG -f models/detect_goggle.xml -g models/face_detector.xml'
```
2. To run 'gather_feature.py'
```
Use command $ 'python3 gather_feature.py -p -n -f -g'
-p is followed by the path to positive images directory.
-n is followed by the path to negative images directory.
-f is followed by the path to positive features directory.
-g is followed by the path to negative features directory.
```
3. To run 'train_svm'
```
Use command $ 'python3 train_svm.py -p -n -s'
-p is followed by the path to positive features directory.
-n is followed by the path to negative features directory.
-s is followed by the name of the model.
```
4. To run 'hard_neg_mining.py'
```
Use command $ 'python3 hard_neg_mining.py -i -p -n -r -m'.
-i is followed by the path to directory contains neg samples.
-p is followed by the path to pos features.
-n is followed by the path to neg features.
-r is followed by the starting round of hard negative mining.
-m is followed by the path to the fist model used in hard negatie training.
```
5. To run 'visualization.py'
```
Use command $ 'python3 visualization.py -s'
-s is followed by the path to source directory containing the images.
```
6. To run 'image_resize.py'
```
Use command $ 'python3 image_resize.py -t -s -f '
-t is followed by the path to store the resized images
-s is followed by the path to a directory contains images to be resized
-f is followed by the option 'flip', default is 'No', give 'Yes' to enable flipping.
as a result of flip mode, any image resized will be flipped and stored again.
```
7. To run 'Image_net_downloader.py'
<br>before running this application, you have to copy all the links of your interest cateogry from ImageNet and Paste them in a 'txt' file, then change the config in python script to modified the path to txt file and the destination of output.</br>
```
Use 'python3 Image_net_downloader.py' directly.

```
8. To run 'test_custom_detector.py'
```
Use 'python3 test_custom_detector.py -i -f -g'
-i is followed by the path to test image.
-f is the path to facial detector
-g is the path to the goggle detctor.
```
