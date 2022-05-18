######## Webcam Object Car Detection  #########
# Credits to : Evan Juras & Ethan Dell
# TensorFlow Models were developed and trained by them
#--------------------------------------------------------------------------
# LIBRARIES
#--------------------------------------------------------------------------
# Import packages
from multiprocessing.connection import wait
import os
import argparse
from turtle import screensize
import cv2
import numpy as np
import sys
import pdb
import time
import pathlib
from threading import Thread
import importlib.util
import datetime
import time
import RPi.GPIO as GPIO

#--------------------------------------------------------------------------
# GLOBAL
#--------------------------------------------------------------------------
GPIO.setmode(GPIO.BCM)

#led for record
GPIO.setup(4, GPIO.OUT)

#led for left
GPIO.setup(11, GPIO.OUT)
#led for right
GPIO.setup(10, GPIO.OUT)

#button
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)


#--------------------------------------------------------------------------
# FUNCTIONS
#--------------------------------------------------------------------------
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])           
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
#--------------------------------------------------------------------------
    def start(self):
        Thread(target=self.update,args=()).start()
        return self
#--------------------------------------------------------------------------
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            # handle next frame
            (self.grabbed, self.frame) = self.stream.read()
#--------------------------------------------------------------------------
    def read(self):
    # Return the most recent frame
        return self.frame
#--------------------------------------------------------------------------
    def stop(self):
        self.stopped = True
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--output_path', help="Where to save processed imges from pi.",
                    required=True)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu
#--------------------------------------------------------------------------
# Import TensorFlow libraries
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# First label is '???', which has to be removed
if labels[0] == '???':
    del(labels[0])
#--------------------------------------------------------------------------
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

led_on = False
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


try:
    print("Progam started - waiting for button push...")
    while True:
        if not led_on and  not GPIO.input(17):
            #timestamp an output directory for each capture
            outdir = pathlib.Path(args.output_path) / time.strftime('%Y-%m-%d_%H-%M-%S-%Z')
            outdir.mkdir(parents=True)
            GPIO.output(4, True)
            time.sleep(.1)
            led_on = True
            f = []

            # Initialize frame rate calculation
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()

            # Initialize video stream
            videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
            time.sleep(1)

            #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
            while True:

                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()

                # Grab frame from video stream
                frame1 = videostream.read()

                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()

                # Retrieve detection results
                boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
                #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                        # Draw label
                        object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                        #Left LED Detection
                        if((scores[i]) > 70 & xmin > screensize/2):
                            GPIO.output(11, True)
                            time.sleep(1)
                            GPIO.output(11, False)
                        #Right LED Detection
                        else:
                            GPIO.output(10, True)
                            time.sleep(1)
                            GPIO.output(10, False)

                # Draw framerate in corner of frame
                cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                # All the results have been drawn on the frame, so it's time to display it.
                #cv2.imshow('Object detector', frame)

                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1
                f.append(frame_rate_calc)

                #path = '/home/pi/tflite1/webcam/' + str(datetime.datetime.now()) + ".jpg"
                path = str(outdir) + '/'  + str(datetime.datetime.now()) + ".jpg"
    
                status = cv2.imwrite(path, frame)


                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q') or led_on and not GPIO.input(17):
                    print(f"Saved images to: {outdir}")
                    GPIO.output(4, False)
                    led_on = False
                    # Clean up
                    cv2.destroyAllWindows()
                    videostream.stop()
                    time.sleep(2)
                    break
finally:
    GPIO.output(4, False)
    GPIO.cleanup()

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    print(str(sum(f)/len(f)))
