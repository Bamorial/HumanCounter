# HumanCounter

## About
This Python script uses computer vision techniques to detect the number of people in an image using HOG and YOLO-based detectors. 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bamorial/HumanCounter

2. Run 
   ```bash
   pip install -r requirements.txt

3. Create a folder called YOLO in this project's root directory 
   ```bash
   mkdir YOLO

4. Download:
    * yolov3.cfg from : https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    * yolov3.weights from: https://pjreddie.com/media/files/yolov3.weights

    Add them to your YOLO folder 

## Overview

   * main.py - main script to execute the people detection
   * IDetector - the abstract class that declares the methods for all detectors
   * HOGDetector - implementation of the HOG+SVM method
   * YOLODetector - implementation for the YOLO method

## Output

1. Text
    * Prints the following message: {{detector.name}} found: n people
   ```bash
   HOGDetector found: 5 people
   YOLODetector found: 4 people

2. Images
    * Shows the original picture with the selectected window
    * Shows one window for each detector, using boxes to show the location of the people found

