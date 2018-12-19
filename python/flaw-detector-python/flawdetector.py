
'''
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
'''

import cv2
import argparse
import socket
import os

import numpy as np

from math import atan2

# Lower and upper value of color Range of the object for color thresholding to detect the object



'''
*********************************************** Orientation Defect detection ******************************************
 Step 1: Convert 3D matrix of contours to 2D 
 Step 2: Apply PCA algorithm to find angle of the data points.
 Step 3: If angle is greater than 0.5, return_flag is made to True else false. 
 Step 4: Save the image in "Orientation" folder if it has a orientation defect.
***********************************************************************************************************************
'''


def get_orientation(contours):
    """
    Gives the angle of the orientation of the object in radians
    :param contours: contour of the object from the frame
    :return: angle of orientation of the object in radians
    """
    size_points = len(contours)
    # data_pts stores contour values in 2D
    data_pts = np.empty((size_points, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = contours[i, 0, 0]
        data_pts[i, 1] = contours[i, 0, 1]
    # Use PCA algorithm to find angle of the data points
    mean, eigenvector = cv2.PCACompute(data_pts, mean=None)
    angle = atan2(eigenvector[0, 1], eigenvector[0, 0])
    return angle


def detect_orientation(frame, contours, base_dir, count_object):
    """
    Identifies the Orientation of the object based on the detected angle
    :param frame: Input frame from video
    :param contours: contour of the object from the frame
    :return: defect_flag, object_defect
    """
    object_defect = "Defect : Orientation"
    # Find the orientation of each contour
    angle = get_orientation(contours)
    # If angle is less than 0.5 then we conclude that no orientation defect is present
    if angle < 0.5:
        defect_flag = False
    else:
        x, y, w, h = cv2.boundingRect(contours)
        print("Orientation defect detected in object {}".format(count_object))
        defect_flag = True
        cv2.imwrite("{}/orientation/Orientation_{}.jpg".format(base_dir, count_object), frame[y - 5: y + h + 10, x - 5: x + w + 10])
    return defect_flag, object_defect


'''
*********************************************** Color Defect detection ************************************************
 Step 1: Increase the brightness of the image
 Step 2: Convert the image to HSV Format. HSV color space gives more information about the colors of the image. 
         It helps to identify distinct colors in the image.
 Step 3: Threshold the image based on the color using "inRange" function. Range of the color, which is considered as 
         a defect for the object, is passed as one of the argument to inRange function to create a mask
 Step 4: Morphological opening and closing is done on the mask to remove noises and fill the gaps
 Step 5: Find the contours on the mask image. Contours are filtered based on the area to get the contours of defective
         area. Contour of the defective area is then drawn on the original image to visualize.
 Step 6: Save the image in "color" folder if it has a color defect.
***********************************************************************************************************************
'''


def detect_color(frame, cnt, base_dir, count_object):
    """
    Identifies the color defect W.R.T the set default color of the object
    :param frame: Input frame from the video
    :param cnt: Contours of the object
    :return: color_flag, object_defect
    """
    LOWER_COLOR_RANGE = (0, 0, 0)
    UPPER_COLOR_RANGE = (174, 73, 255)

    object_defect = "Defect : Color"
    color_flag = False
    # Increase the brightness of the image
    cv2.convertScaleAbs(frame, frame, 1, 20)
    # Convert the captured frame from BGR to HSV
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold the image
    img_thresholded = cv2.inRange(img_hsv, LOWER_COLOR_RANGE, UPPER_COLOR_RANGE)
    # Morphological opening (remove small objects from the foreground)
    img_thresholded = cv2.erode(img_thresholded, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    img_thresholded = cv2.dilate(img_thresholded, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, hierarchy = cv2.findContours(img_thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if 2000 < area < 10000:
            cv2.drawContours(frame, contours[i], -1, (0, 0, 255), 2)
            color_flag = True
    if color_flag == True:
        x,y,w,h =cv2.boundingRect(cnt)
        print("Color defect detected in object {}".format(count_object))
        print("{}/color/Color_{}.jpg".format(base_dir, count_object))
        cv2.imwrite("{}/color/Color_{}.jpg".format(base_dir, count_object), frame[y - 5: y + h + 10, x - 5: x + w + 10])
    return color_flag, object_defect


'''
**************************************************** Crack detection **************************************************
 Step 1: Convert the image to gray scale
 Step 2: Blur the gray image to remove the noises
 Step 3: Find the edges on the blurred image to get the contours of possible cracks
 Step 4: Filter the contours to get the contour of the crack
 Step 5: Draw the contour on the orignal image for visualization
 Step 6: Save the image in "crack" folder if it has crack defect
***********************************************************************************************************************
'''


def detect_crack(frame, cnt, base_dir, count_object):
    """
    Identifies the Crack defect on the object
    :param frame: Input frame from the video
    :param cnt: Contours of the object
    :return: defect_flag, object_defect, cnt
    """
    object_defect = "Defect : Crack"
    defect_flag = False
    low_threshold = 130
    kernel_size = 3
    ratio = 3
    # Convert the captured frame from BGR to GRAY
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (7, 7))
    # Find the edges
    detected_edges = cv2.Canny(img, low_threshold, low_threshold * ratio, kernel_size)
    # Find the contours
    contours, hierarchy = cv2.findContours(detected_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = []
    if len(contours) != 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 20 or area < 9:

                cv2.drawContours(frame, contours, i, (0, 255, 0), 2)
                defect_flag = True
                cnts.append(contours[i])

        if defect_flag == True:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.imwrite("{}/crack/Crack_{}.jpg".format(base_dir, count_object), frame[y - 5: y + h + 20, x - 5: x + w + 10])
            print("Crack defect detected in object {}".format(count_object))
    return defect_flag, object_defect, cnt



def runFlowDetector(vid_path = 0, base_dir=None, draw_callback = None):

    OBJECT_AREA = 9000
    LOW_H = 0
    LOW_S = 0
    LOW_V = 47
    HIGH_H = 179
    HIGH_S = 255
    HIGH_V = 255
    count_object = 0


    if base_dir == None:
       base_dir = os.getcwd()


    dir_names = ["crack", "color", "orientation", "no_defect"]
    frame_count = 0
    num_of_dir = 4
    frame_number = 40
    defect = []

    # create folders with the given dir_names to save defective objects
    for i in range(len(dir_names)):
        if not os.path.exists(os.path.join(base_dir, dir_names[i])):
            os.makedirs(os.path.join(base_dir, dir_names[i]))
        else:
            file_list = os.listdir(os.path.join(base_dir, dir_names[i]))
            for f in file_list:
                os.remove(os.path.join(base_dir,dir_names[i],f))

    capture = cv2.VideoCapture(vid_path)

    # Check if video is loaded successfully
    if capture.isOpened():
        print("Opened video!!")

    else:
        print("Problem loading the video!!!")

    # OpenCV video write to store the output video
    try:
        vw = None
        while True:
            # Read the frame from the stream
            _, frame = capture.read()

            if np.shape(frame) == ():
                break

            frame_count += 1
            object_count = "Object Number : {}".format(count_object)

            # Check every given frame number (Number chosen based on the frequency of object on conveyor belt)
            if frame_count % frame_number == 0:
                defect = []

                # Convert BGR image to HSV color space
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Thresholding of an Image in a color range
                img_thresholded = cv2.inRange(img_hsv, (LOW_H, LOW_S, LOW_V), (HIGH_H, HIGH_S, HIGH_V))

                # Morphological opening(remove small objects from the foreground)
                img_thresholded = cv2.erode(img_thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                img_thresholded = cv2.dilate(img_thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

                # Morphological closing(fill small holes in the foreground)
                img_thresholded = cv2.dilate(img_thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                img_thresholded = cv2.erode(img_thresholded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

                # Find the contours on the image
                contours, hierarchy = cv2.findContours(img_thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                for cnt in contours:
                    x, y, width, height = cv2.boundingRect(cnt)
                    if width * height > OBJECT_AREA:
                        count_object += 1
                        frame_copy = frame.copy()
                        object_count = "Object Number : {}".format(count_object)

                        # Check for the orientation of the object
                        orientation_flag, orientation_defect = detect_orientation(frame, cnt, base_dir, count_object)
                        if orientation_flag == True:
                            defect.append(str(orientation_defect))

                        # Check for the color defect of the object
                        color_flag, color_defect = detect_color(frame, cnt, base_dir, count_object)
                        if color_flag == True:
                            defect.append(str(color_defect))

                        # Check for the crack defect of the object
                        crack_flag, crack_defect, crack_contour = detect_crack(frame_copy, cnt, base_dir, count_object)
                        if crack_flag == True:
                            defect.append(str(crack_defect))
                            cv2.drawContours(frame, crack_contour, -1, (0, 255, 0), 2)

                        # Check if none of the defect is found
                        if not defect:
                            object_defect = "No Defect"
                            defect.append(str(object_defect))
                            print("No defect detected in object {}".format(count_object))
                            cv2.imwrite("{}/no_defect/Nodefect_{}.jpg".format(base_dir, count_object), frame[y - 5: y + height + 10, x - 5: x + width + 10])
                if not defect:
                    continue

            all_defects = " ".join(defect)

            cv2.putText(frame, all_defects, (5, 130), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, object_count, (5, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            if draw_callback != None:
                draw_callback(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if vw == None:
                height = np.size(frame, 0)
                width = np.size(frame, 1)
                out_dir = os.path.join(base_dir, 'inference_output.mp4')
                vw = cv2.VideoWriter(out_dir, 0x00000021, 15.0, (width, height), True)
            vw.write(frame)
    finally:
        if vw != None:
            vw.release()


if __name__ == '__main__':
    data = []
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=False, help="Name of the directory to which defective images are saved")
    parser.add_argument('-f', '--vid', default=0, help="Name of the video file")
    args = vars(parser.parse_args())
    base_dir = args['dir']
    vid_path = args['vid']
    runFlowDetector(vid_path, base_dir)


