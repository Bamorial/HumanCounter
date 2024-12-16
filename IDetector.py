from abc import ABC, abstractmethod

class IDetector:
    @abstractmethod
    def initiate(image):
        pass
    @abstractmethod
    def show_image(image):
        pass
    @abstractmethod
    def get_number_of_people(image):
        pass