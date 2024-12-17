import cv2
from Window import Window
from Point import Point
from HOGDetector import HOGDetector
from YOLODetector import YOLODetector
import numpy as np
import imutils

highlight_color=(0,0,255)
path_to_image='./demo/demo6.jpg'

def Crop(image, window: Window):
    cropped_image=np.zeros((window.height,window.width,3), dtype=np.uint8)
    x,y = window.corner.x, window.corner.y
    width=window.width
    height=window.height
    cropped_image=image[y:y+height,x:x+width]
    return cropped_image

def Resize(image, width):
    image=imutils.resize(image,width) #I use this to maintain the aspect ratio of the photo
    return image

def ShowSelection(input_image ,window):
    x,y = window.corner.x, window.corner.y
    width=window.width
    height=window.height
    for i in range(len(input_image)):
        for j in range(len(input_image[i])):
            if (i == y or i == y + height - 1) and x <= j < x + width:
                input_image[i, j] = highlight_color
            if (j == x or j == x + width - 1) and y <= i < y + height:
                input_image[i, j] = highlight_color
    cv2.imshow('Selected Window', input_image)

def main():
    image=cv2.imread(path_to_image)
    window=Window(
        corner=Point(100,0), 
        width=700, 
        height=500)
    cropped_image= Crop(image, window) 
    cv2.imshow('Image after crop',cropped_image)
    ShowSelection(image, window)
    resized_image=Resize(cropped_image, 1000)
    detectors=[HOGDetector(), YOLODetector()]
    for detector in detectors:
        detector.initiate(resized_image)
        number_of_people=detector.get_number_of_people()
        detector.show_image()
        print(detector.name+' found: '+str(number_of_people)+' people')
    cv2.waitKey(0)

main()