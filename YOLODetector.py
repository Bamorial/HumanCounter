from IDetector import IDetector
import numpy as np
import cv2

class YOLODetector(IDetector):
    name="YOLODetector"
    __config_path='./YOLO/yolov3.cfg'
    __weights_path='./YOLO/yolov3.weights'
    __labels_path="./coco.names"

    def initiate(self, image):
        self.image=image
        with open(self.__labels_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNetFromDarknet(self.__config_path, self.__weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def get_number_of_people(self):
        (h, w) = self.image.shape[:2]
        blob = cv2.dnn.blobFromImage(self.image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        self.boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == "person":  # Detect "person"
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # Calculate top-left corner
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    self.boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression to reduce overlapping boxes
        self.indices = cv2.dnn.NMSBoxes(self.boxes, confidences, 0.5, 0.4)
        return len(self.indices)

    def show_image(self):
        for i in self.indices:
            (x, y, w, h) = self.boxes[i]
            color = (0, 255, 0)  # Green for humans
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("Detectie Oameni metoda YOLO", self.image)
