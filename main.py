import cv2
from Window import Window
from Point import Point
from HOGDetector import HOGDetector
from YOLODetector import YOLODetector
import numpy as np
from utils import Crop, Resize, ShowSelection

path_to_image='./demo/demo3.jpg'


def main():
    image=cv2.imread(path_to_image)
    window=Window(
        corner=Point(600,0), 
        width=1000, 
        height=1000)
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