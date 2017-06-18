from skimage.feature import hog
from skimage import color
from imutils.object_detection import non_max_suppression
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib
import numpy as np
import cv2
import imutils



def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size
    equal to 'window_size'. The first image returned top-left
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def detect_goggle(im):
    print('Begin to detect the facial area %d %d' %(im.shape[1],im.shape[0]))
    min_wdw_sz = (128, 64)
    step_size = (15, 15)    # 注意调整
    downscale = 1.15   #注意调整  default= 1.15

    clf = joblib.load('/Users/nick/Dropbox/Python/PythonWorkplace/Computer_vision/goggle_detection/train4_hard_round_6.model')

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0

    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue

            im_window = color.rgb2gray(im_window)
            im_window.astype(np.uint8)
            #scikit learn detector
            fd = hist=hog(im_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
            fd = fd.reshape(1, -1)

            pred = clf.predict(fd)

            ##if this window contains a people:
            if pred == 1:

                if clf.decision_function(fd) > 0.8:
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf.decision_function(fd),
                         int(min_wdw_sz[0] * (downscale ** scale)),
                         int(min_wdw_sz[1] * (downscale ** scale))))

        scale += 1

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)  # original probs=sc overlapThresh=0.3
    # print("shape, ", pick.shape)

    return pick
    # for (xA, yA, xB, yB) in pick:
    #     cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)






face_cascade =cv2.CascadeClassifier('/Users/nick/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('/Users/nick/desktop/11.jpeg')

print('original shape: %d x %d'%(img.shape[1],img.shape[0]))

im = imutils.resize(img, width=min(2400, img.shape[1]))

print('resized to %d x %d'%(im.shape[1],im.shape[0]))

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.05, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(im,(x-int(w*0.3),y-int(h*0.3)),(x+w+int(w*0.3),y+h+int(h*0.3)),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]   # the area that is identified as FACE
    roi_color = im[y-int(h*0.3):y+h+int(h*0.3), x:x+w+int(w*0.3)]   # the area that is identified as FACE

    cv2.imshow('11',im)
    cv2.waitKey(0)

    goggle_result_pick=detect_goggle(roi_color)

    print(goggle_result_pick)
    #detect goggle in the facial areas

    for (xA, yA, xB, yB) in goggle_result_pick:
        cv2.rectangle(im,(x+xA,y+yA),(x+xB,y+yB),(0,255,0),2)

    # cv2.imshow('area',roi_gray)
    # print('This facial are has a shape of %d x %d' %(w,h) )
    # cv2.waitKey(0)

cv2.imshow('img',imutils.resize(im,width=600))
cv2.waitKey(0)
cv2.destroyAllWindows()
