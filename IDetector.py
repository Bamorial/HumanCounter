from abc import ABC, abstractmethod

class IDetector:
    
    @abstractmethod
    def get_number_of_people(image):
        pass