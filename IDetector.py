from abc import ABC, abstractmethod

class IDetector:
    @abstractmethod #i use this because i want to maintain a structure for all the classes that implement this "interface"
    def initiate(image):
        pass
    @abstractmethod
    def show_image():
        pass
    @abstractmethod
    def get_number_of_people():
        pass