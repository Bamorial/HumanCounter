from IDetector import IDetector

class YOLODETECTOR(IDetector):
    name="YOLODetector"
    __config_path='./YOLO/yolov3.config'
    __weights__path='./YOLO/yolov3.weights'
    __labels_path="./YOLO/coco.names"

    def initiate(self, image):
        return 0

    def get_number_of_people(self):
        return 0

    def show_image(self,image):
        pass
