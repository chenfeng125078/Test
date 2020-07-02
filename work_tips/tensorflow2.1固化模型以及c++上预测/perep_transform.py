from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# 四点透视变换
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#    help="path to the input image")
# args = vars(ap.parse_args())
# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread("image/1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)
# find contours in edge map ,then initialize the contour corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("a", edged)
cv2.waitKey(0)
# print(cnts)
print("----------------", len(cnts))
cnts = cnts[0]
docCnt = None
# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print(approx)
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break
# docCnt = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])
print(type(docCnt))

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper

paper = four_point_transform(image, docCnt.reshape(4, 2))
# warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv2.imshow("original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
