from IDetector import IDetector
import cv2
import copy

class HOGDetector(IDetector):
    name='HOGDetector'
    color=(255,0,0)

    def initiate(self, image):
        self.image=image
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(image, winStride=(6,6))
        self.boxes=boxes
        self.weights=weights

    def get_number_of_people(self):
        return len(self.boxes)

    def show_image(self):
        work_image=copy.deepcopy(self.image)
        for (x, y, w, h) in self.boxes:
            cv2.rectangle(work_image, (x, y), (x + w, y + h), self.color, 2)
        cv2.imshow("Detectie Oameni Metoda HOG", work_image)
