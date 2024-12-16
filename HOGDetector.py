from IDetector import IDetector
import cv2

class HOGDetector(IDetector):
    name='HOGDetector'

    def initiate(self, image):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        self.boxes=boxes
        self.weights=weights



    def get_number_of_people(self):
        return len(self.boxes)

    def show_image(self,image):
        for (x, y, w, h) in self.boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detectare Oameni", image)
        cv2.waitKey(0)
