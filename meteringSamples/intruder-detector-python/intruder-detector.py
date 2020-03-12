
'''
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy
import time
import collections
import signal
import pathlib
from pathlib import Path
from inference import Network
from qarpo.demoutils import *
import applicationMetricWriter


# CONSTANTS
CONF_FILE = "./resources/conf.txt"
TARGET_DEVICE = "CPU"
CPU_EXTENSION = ""
LOOP_VIDEO = False
CONF_THRESHOLD_VALUE = 0.55
LOG_FILE_PATH = "./intruders.log"
LOG_WIN_HEIGHT = 432
LOG_WIN_WIDTH = 410
CONF_CANDIDATE_CONFIDENCE = 4
CODEC = cv2.VideoWriter_fourcc(*'avc1')
output_dir=" "

# Global variables
model_xml = ''
model_bin = ''
conf_labels_file_path = ''
video_caps = []


# Event class to store the intruder details
class Event:
    def __init__(self, event_time=None, intruder=None, count=None, frame=None):
        self.time = event_time
        self.intruder = intruder
        self.count = count
        self.frame = frame


# VideoCap class to manage the input source
class VideoCap:
    def __init__(self, vc, cam_name, cams, is_cam):
        self.input_width = vc.get(3)
        self.input_height = vc.get(4)
        self.vc = vc
        self.cam_name = cam_name
        self.is_cam = is_cam
        self.no_of_labels = 0
        self.last_correct_count = []
        self.total_count = []
        self.current_count = []
        self.changed_count = []
        self.candidate_count = []
        self.candidate_confidence = []
        self.frame = None
        self.loop_frames = 0
        self.frame_count = 0
        self.events = []
        self.video_name = 'video{}.mp4'.format(cams)
        self.vw = None
        
    def init(self, size):
        self.no_of_labels = size
        for i in range(size):
            self.last_correct_count.append(0)
            self.total_count.append(0)  
            self.changed_count.append(False)
            self.current_count.append(0)
            self.candidate_count.append(0)
            self.candidate_confidence.append(0)

    def init_vw(self, h, w):
        self.vw = cv2.VideoWriter(os.path.join(output_dir, self.video_name), CODEC,
                                  self.vc.get(cv2.CAP_PROP_FPS), (w,h), True)
        if not self.vw.isOpened():
            return -1, self.video_name
        return 0, ''


def parse_args():
    """
    Parse the command line argument

    :return None:
    """
    global LOOP_VIDEO
    global TARGET_DEVICE
    global conf_labels_file_path
    global model_xml
    global model_bin
    global CPU_EXTENSION
    global output_dir
    
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model's weights.", 
                        required=True, type=str)
    parser.add_argument("-lb", "--labels", help="Labels mapping file", default=None, 
                        type=str, required=True)
    parser.add_argument("-d", "--device", help="Device to run the inference (CPU, GPU , FPGA or MYRIAD). Default option is CPU.",
                        required=False, type=str)
    parser.add_argument("-lp", "--loop", help="Loop video to mimic continuous input.", type=str, default=None)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels "
                        "impl.", type=str, default=None)
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)

    args = parser.parse_args()
    if args.model:
        model_xml = args.model
    if args.labels:
        conf_labels_file_path = args.labels
    if args.device:
        TARGET_DEVICE = args.device
    if args.loop:
        if args.loop == "True" or args.loop == "true":
            LOOP_VIDEO = True
        elif args.loop == "False" or args.loop == "false":
            LOOP_VIDEO = False     
        else:
            print("Invalid input for -lp/--loop. Defaulting to LOOP_VIDEO = False")     
            LOOP_VIDEO = False
    if args.cpu_extension:
        CPU_EXTENSION = args.cpu_extension
    if args.output_dir:
        output_dir=args.output_dir

def get_used_labels(req_labels):
    """
    Read the model's label file and get the position of labels required by the application

    :param req_labels: intruders to be detected in the input source
    :return status: 0 on success, negative value on failure
            labels: On success, list of labels present in model's label file
            used_labels: On success, list of bool values where true indicates that a label in labels list at that position is
                    used in the application
    """
    global conf_labels_file_path
    used_labels = []

    if conf_labels_file_path:
        labels = []
        with open(conf_labels_file_path, 'r') as label_file:
            if not label_file:
                return [-4, [], []]
            labels = [x.strip() for x in label_file]

        if not labels:
            return [-5, [], []]

        for label in labels:
            if label in req_labels:
                used_labels.append(True)
            else:
                used_labels.append(False)

        return [0, labels, used_labels]

    return [-6, [], []]


def get_input():
    """
    Parse the configuration file

    :return status: 0 on success, negative value on failure
            streams: On success, list of VideoCap containing configuration file data
            labels: On success, labels or intruder to be detected
    """
    global CONF_FILE
    global video_caps
    cams = 0
    labels = []
    streams = []
    file = open(CONF_FILE, 'r')
    if not file:
        return [-7, '']

    file_data = file.readlines()

    for line in file_data:
        words = line.split()
        if len(words) is 0:
            continue
        if words[0] == 'video:':
            cams += 1
            cam_name = "Cam {}".format(cams)
            if words[1].isdigit():
                video_cap = VideoCap(cv2.VideoCapture(int(words[1])), cam_name, cams, is_cam=True)
            else:
                if os.path.isfile(words[1]):
                    video_cap = VideoCap(cv2.VideoCapture(words[1]), cam_name, cams, is_cam=False)
                else:
                    return [-8, [words[1]]]
            video_caps.append(video_cap)
        elif words[0] == 'intruder:':
                labels.append(words[1])
        else:
            print("Unrecognized option; Ignoring")
    
    for video_cap in video_caps:
        if not video_cap.vc.isOpened():
            return [-9, [video_cap.cam_name]]

        video_cap.init(len(labels))
    file.close()
    return [0, labels]

def clean_up():
    """
    Destroys all the opencv windows and releases the objects of videoCapture and videoWriter
    """
    global video_caps
    for video_cap in video_caps:
        if video_cap.vw:
            video_cap.vw.release()
        if video_cap.vc:
            video_cap.vc.release()


def intruder_detector():
    """
    Process the input source frame by frame and detects intruder, if any.

    :return status: 0 on success, negative value on failure
    """
    global CONF_CANDIDATE_CONFIDENCE
    global LOG_WIN_HEIGHT
    global LOG_WIN_WIDTH
    global CONF_FILE
    global video_caps
    global conf_labels_file_path

    parse_args()
    if not os.path.isfile(CONF_FILE):
        return -12, ""
    
    if not os.path.isfile(conf_labels_file_path):
        return -13, ""

    # Creates subdirectory to save output snapshots
    pathlib.Path(os.getcwd() + '/output/').mkdir(parents=True, exist_ok=True)

    # Read the configuration file
    ret, req_labels = get_input()
    if ret != 0:
        return ret, req_labels[0]

    if not video_caps:
        return -14, ''

    # Get the labels that are used in the application
    ret, label_names, used_labels = get_used_labels(req_labels)
    if ret != 0:
        return ret, ''
    if True not in used_labels:
        return -15, ''

    # Init a rolling log to store events
    rolling_log_size = int((LOG_WIN_HEIGHT - 15) / 20)
    log_list = collections.deque(maxlen=rolling_log_size)

    # Open a file for intruder logs
    log_file = open(LOG_FILE_PATH, 'w')
    if not log_file:
        return -16, ''
    
    # Initializing VideoWriter for each source
    for video_cap in video_caps:
        
        ret, ret_value = video_cap.init_vw(int(video_cap.input_height), int(video_cap.input_width))
        if ret != 0:
            return ret, ret_value
    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(model_xml,TARGET_DEVICE, 1, 1, 0, CPU_EXTENSION)

    min_fps = min([i.vc.get(cv2.CAP_PROP_FPS) for i in video_caps])
    no_more_data = [False] * len(video_caps)
    start_time = time.time()
    inf_time = 0
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    statsVideo = cv2.VideoWriter(os.path.join(output_dir,'Statistics.mp4'), fourcc, min_fps, (LOG_WIN_WIDTH, LOG_WIN_HEIGHT), True)
    job_id = os.environ['PBS_JOBID']
    progress_file_path = os.path.join(output_dir,'i_progress_'+str(job_id)+'.txt')
    infer_start_time = time.time() 
    # Main loop starts here. Loop over all the video captures
    while True:
        for idx, video_cap in enumerate(video_caps):
            # Get a new frame
            vfps = int(round(video_cap.vc.get(cv2.CAP_PROP_FPS)))
            for i in range(0, int(round(vfps / min_fps))):
                ret, video_cap.frame = video_cap.vc.read()
                video_cap.loop_frames += 1
                # If no new frame or error in reading a frame, exit the loop
                if not ret:
                    no_more_data[idx] = True
                    break
            if no_more_data[idx]:
                stream_end_frame = numpy.zeros((int(video_cap.input_height), int(video_cap.input_width), 1),
                                              dtype='uint8')
                stream_end_message = "Stream from {} has ended.".format(video_cap.cam_name)
                cv2.putText(stream_end_frame, stream_end_message, (int(video_cap.input_width/2) - 30,
                            int(video_cap.input_height/2) - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                continue
            for i in range(video_cap.no_of_labels):
                video_cap.current_count[i] = 0
                video_cap.changed_count[i] = False


            # Resize to expected size (in model .xml file)
            # Input frame is resized to infer resolution
            in_frame = cv2.resize(video_cap.frame, (w, h))

            # PRE-PROCESS STAGE:
            # Convert image to format expected by inference engine
            # IE expects planar, convert from packed
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))  
            in_frame = in_frame.reshape((n, c, h, w))
            # Start asynchronous inference for specified request.
            inf_start = time.time()
            infer_network.exec_net(0, in_frame)
            # Wait for the result
            if infer_network.wait(0) == 0:
                inf_time = time.time() - inf_start
                applicationMetricWriter.send_inference_time(inf_time*1000)
                # Results of the output layer of the network
                res = infer_network.get_output(0)
                for obj in res[0][0]:
                    label = int(obj[1]) - 1
                    # Draw the bounding box around the object when the probability is more than specified threshold
                    if obj[2] > CONF_THRESHOLD_VALUE and used_labels[label]:                   
                        video_cap.current_count[label] += 1
                        xmin = int(obj[3] * video_cap.input_width)
                        ymin = int(obj[4] * video_cap.input_height)
                        xmax = int(obj[5] * video_cap.input_width)
                        ymax = int(obj[6] * video_cap.input_height)
                        # Draw bounding box around the intruder detected
                        cv2.rectangle(video_cap.frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4, 16)      

                for i in range(video_cap.no_of_labels): 
                    if video_cap.candidate_count[i] == video_cap.current_count[i]:
                        video_cap.candidate_confidence[i] += 1
                    else:
                        video_cap.candidate_confidence[i] = 0
                        video_cap.candidate_count[i] = video_cap.current_count[i]

                    if video_cap.candidate_confidence[i] == CONF_CANDIDATE_CONFIDENCE:
                        video_cap.candidate_confidence[i] = 0
                        video_cap.changed_count[i] = True
                    else:
                        continue

                    if video_cap.current_count[i] > video_cap.last_correct_count[i]:
                        video_cap.total_count[i] += video_cap.current_count[i] - video_cap.last_correct_count[i]
                        det_objs = video_cap.current_count[i] - video_cap.last_correct_count[i]
                        total_count = sum(video_cap.total_count)
                        for det_obj in range(det_objs):
                            current_time = time.strftime("%H:%M:%S")
                            log = "{} - Intruder {} detected on {}".format(current_time, label_names[i],
                                                                           video_cap.cam_name)
                            print(log)
                            log_list.append(log)
                            log_file.write(log + "\n")
                            event = Event(event_time=current_time, intruder=label_names[i], count=total_count,
                                          frame=video_cap.frame_count)
                            video_cap.events.append(event)
                            
                        snapshot_name = "output/intruder_{}.png".format(total_count)
                        cv2.imwrite(snapshot_name, video_cap.frame)
                    video_cap.last_correct_count[i] = video_cap.current_count[i]
            # Create intruder log window, add logs to the frame and display it
            log_window = numpy.zeros((LOG_WIN_HEIGHT, LOG_WIN_WIDTH, 1), dtype='uint8') 
            for i, log in enumerate(log_list):
                cv2.putText(log_window, log, (10, 20 * i + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            log_window = cv2.cvtColor(log_window, cv2.COLOR_GRAY2BGR)
            statsVideo.write(log_window)
            video_cap.frame_count += 1

            # Video output        
            inf_time_message = "Inference time: {:.3f} ms".format(inf_time * 1000)
            cv2.putText(video_cap.frame, inf_time_message, (10, int(video_cap.input_height) - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            fps_time = time.time() - start_time
            fps_message = "FPS: {:.3f} fps".format(1/fps_time)
            cv2.putText(video_cap.frame, fps_message, (10, int(video_cap.input_height) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # Display the video output
            video_cap.vw.write(video_cap.frame)
            if video_cap.frame_count %10 == 0: 
                progressUpdate(progress_file_path, time.time()-infer_start_time,video_cap.frame_count, int(video_cap.vc.get(cv2.CAP_PROP_FRAME_COUNT)))
            start_time = time.time()

            # Loop video to mimic continuous input if LOOP_VIDEO flag is True
            if LOOP_VIDEO and not video_cap.is_cam:
                vfps = int(round(video_cap.vc.get(cv2.CAP_PROP_FPS)))
                # If a video capture has ended restart it
                if video_cap.loop_frames > video_cap.vc.get(cv2.CAP_PROP_FRAME_COUNT) - int(round(vfps / min_fps)):
                    video_cap.loop_frames = 0
                    video_cap.vc.set(cv2.CAP_PROP_POS_FRAMES, 0)      

        if False not in no_more_data:
            progressUpdate(progress_file_path, time.time()-infer_start_time, int(video_cap.vc.get(cv2.CAP_PROP_FRAME_COUNT)), int(video_cap.vc.get(cv2.CAP_PROP_FRAME_COUNT))) 
            break

    no_more_data = False
    t2 = time.time()-infer_start_time
    for videos in video_caps:
   	 with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        	f.write('{} \n'.format(round(t2)))
        	f.write('{} \n'.format(videos.frame_count))

    infer_network.clean()
    log_file.close()
    return 0, ''


if __name__ == '__main__':
    status, value = intruder_detector()

    if status == 0:
        print("Success!")
    elif status == -1:
        print("Could not open for write" + value + "!")
    elif status == -2:
        print("You need to specify the path to the .xml file!")
        print("Use -m MODEL or --model MODEL")
    elif status == -3:
        print("You need to specify the path to the labels file!")
        print("Use -l LABELS or --labels LABELS")
    elif status == -4:
        print("Error in opening labels file!")
    elif status == -5:
        print("No labels found in label file!")
    elif status == -6:
        print("Labels file not found!")
    elif status == -7:
        print("Error in opening Configuration file " + CONF_FILE+ "!")
    elif status == -8:
        print("Could not find the video file " + value+ "!")
    elif status == -9:
        print("\nCould not open " + value +" for reading!")
    elif status == -12:
        print(CONF_FILE + " configuration file not found!")
    elif status == -13:
        print(conf_labels_file_path + " label file not found!")
    elif status == -14:
        print("No input source found in configuration file!")
    elif status == -15:
        print("Error: No labels currently in use. Please edit " + CONF_FILE+" file!")
    elif status == -16:
        print("Error in opening intruder log file!")
    else:
        print("Unknown error occurred!")

    clean_up()
    applicationMetricWriter.send_application_metrics(model_xml, TARGET_DEVICE)
