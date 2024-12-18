# HumanCounter


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

## Output

1. Text
    * Prints the following message: {{detector.name}} found: n people
2. Images
    * Shows the original picture with the selectected window
    * Shows one window for each detector, using boxes to show the location of the people found

