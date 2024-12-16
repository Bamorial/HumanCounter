import cv2
from Window import Window
from Point import Point
from HOGDetector import HOGDetector
import numpy as np
import imutils

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

def main():
    image=cv2.imread('./demo/demo1.jpg')
    window=Window(
        corner=Point(0,0), 
        width=300, 
        height=300)
    # image=Resize(image)
    cropped_image= Crop(image, window) 

    cv2.imshow('Image after crop',cropped_image)
    cv2.waitKey(0)

    resized_image=Resize(cropped_image, 1000)

    cv2.imshow('Image after resize',resized_image)
    cv2.waitKey(0)

    detector=HOGDetector()
    detector.initiate(resized_image)
    number_of_people=detector.get_number_of_people()
    detector.show_image(resized_image)
    print(str(number_of_people))

main()