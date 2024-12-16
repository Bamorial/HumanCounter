import cv2
from Window import Window
from Point import Point
from HOGDetector import HOGDetector

def Crop(image, window: Window):
    pass

def main():
    image=cv2.imread('./demo/demo1.jpg')
    detector=HOGDetector()
    print(str(detector.get_number_of_people(image)))
    window=Window(
        corner=Point(0,0), 
        width=10, 
        height=10)
    cropped_image= Crop(image, window) 
    # detectors=[]
    # number_of_people=0
    # for i in detectors:
    #     number_of_people=i.get_number_of_people(image)
    #     print(i.name+': '+ str(number_of_people))

main()