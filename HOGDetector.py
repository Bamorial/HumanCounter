from IDetector import IDetector
import cv2

class HOGDetector(IDetector):
    name='HOGDetector'

    def initiate(self, image):
        self.image=image
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(image, winStride=(12,12))
        self.boxes=boxes
        self.weights=weights

    def get_number_of_people(self):
        return len(self.boxes)

    def show_image(self):
        for (x, y, w, h) in self.boxes:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detectie Oameni Metoda HOG", self.image)
