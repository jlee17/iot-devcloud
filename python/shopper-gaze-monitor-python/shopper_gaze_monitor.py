"""Shopper Gaze Monitor."""

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import json
import time
import cv2
import numpy

from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network
from pathlib import Path
import logging as log

sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import *

# shoppingInfo contains statistics for the shopping information
MyStruct = namedtuple("shoppingInfo", "shopper, looker")
INFO = MyStruct(0, 0)

POSE_CHECKED = False

DELAY = 5


def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Path to an .xml file with a pre-trained"
                             "face detection model")
    parser.add_argument("-pm", "--posemodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                             "head pose model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                             "'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                             "for a suitable plugin for device specified"
                             "(CPU by default)")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", help="Path to output directory", type=str, default=None)
    parser.add_argument('-nireq', '--number_infer_requests',
                        help='Number of parallel inference requests (default is 2).',
                        type=int,
                        required=False,
                        default=2)

    return parser


def face_detection(frame_count, res, args, initial_wh, result_file):  ### (res, args, initial_wh)
    """
    Parse Face detection output.
    :param res: Detection results
    :param args: Parsed arguments
    :param initial_wh: Initial width and height of the FRAME
    :return: Co-ordinates of the detected face
    """
    global INFO
    faces = []
    INFO = INFO._replace(shopper=0)

    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > args.confidence:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])

            faces.append([xmin, ymin, xmax, ymax])
            INFO = INFO._replace(shopper=len(faces))

    return faces

def writeResults(frame_count, args, res, face_infer_time, head_pose_infer_time, shopper, looker, result_file):
    for obj in res[0][0]:
        dims = ""
       # Draw only objects when probability more than specified threshold
        if obj[2] > args.confidence:
            dims = "{frame_id} {face_infer_time} {head_pose_infer_time} {shopper} {looker} \n".format(frame_id=frame_count, face_infer_time=face_infer_time, head_pose_infer_time=head_pose_infer_time, shopper=shopper, looker=looker)

            result_file.write(dims)

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    global INFO
    global DELAY
    global POSE_CHECKED

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logger = log.getLogger()

    # if args.input == 'cam':
    # input_stream = 0
    # else:
    input_stream = args.input
    assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    job_id = os.environ['PBS_JOBID']
    progress_file_path = os.path.join(args.output_dir, 'i_progress_' + job_id + '.txt')
    result_file = open(os.path.join(args.output_dir, 'output_'+job_id+'.txt'), "w")

    if input_stream:
        cap.open(args.input)
        # Adjust DELAY to match the number of FPS of the video file
        DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        return

    # Initialise the class
    infer_network = Network()
    infer_network_pose = Network()
    # Load the network to IE plugin to get shape of input layer
    plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network.load_model(args.model,
                                                                args.device, 1, 1,
                                                                args.number_infer_requests,
                                                                args.cpu_extension)
    n_hp, c_hp, h_hp, w_hp = infer_network_pose.load_model(args.posemodel,
                                                           args.device, 1, 3,
                                                           args.number_infer_requests,
                                                           args.cpu_extension, plugin)[1]

    ret, frame = cap.read()
    infer_time_start = time.time()

    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    infer_requests = infer_network.net_plugin.requests

    current_inference_pose = 0
    previous_inference_pose = 1 - args.number_infer_requests
    infer_requests_pose = infer_network_pose.net_plugin.requests

    frame_count = 0
    ret, frame = cap.read()
    infer_time_start = time.time()
    while frame_count < video_len:
        looking = 0
        ret, next_frame = cap.read()
        if ret:
            initial_wh = [cap.get(3), cap.get(4)]
            in_frame_fd = cv2.resize(next_frame, (w_fd, h_fd))
            # Change data layout from HWC to CHW
            in_frame_fd = in_frame_fd.transpose((2, 0, 1))
            in_frame_fd = in_frame_fd.reshape((n_fd, c_fd, h_fd, w_fd))

            # Start asynchronous inference for specified request
            inf_start_fd = time.time()
            infer_network.net_plugin.start_async(request_id=current_inference, inputs={infer_network.input_blob: in_frame_fd})

        det_time_fd = time.time() - inf_start_fd

        faces = list()

        res = numpy.zeros(1400)
        res = numpy.reshape(res, newshape=(1, 1, 200, 7))
        det_time_hp_pose = 0

        if previous_inference >= 0:
            infer_requests[previous_inference].wait()
            res = infer_requests[previous_inference].outputs[infer_network.out_blob]
            frame_count += 1
            # Results of the output layer of the network
            # Parse face detection output
            faces = face_detection(frame_count, res, args, initial_wh, result_file)

        if len(faces) != 0:
            # Look for poses
            for res_hp in faces:
                if ret:
                    xmin, ymin, xmax, ymax = res_hp
                    head_pose = frame[ymin:ymax, xmin:xmax]
                    in_frame_hp_pose = cv2.resize(head_pose, (w_hp, h_hp))
                    in_frame_hp_pose = in_frame_hp_pose.transpose((2, 0, 1))
                    in_frame_hp_pose = in_frame_hp_pose.reshape((n_hp, c_hp, h_hp, w_hp))

                    inf_start_hp_pose = time.time()

                    infer_network_pose.net_plugin.start_async(request_id=current_inference_pose,
                                                            inputs={infer_network_pose.input_blob: in_frame_hp_pose})
                det_time_hp_pose = time.time() - inf_start_hp_pose

                if previous_inference_pose >= 0:
                    infer_requests_pose[previous_inference_pose].wait()
                # Parse inference results
                angle_p_fc = infer_requests_pose[previous_inference_pose].outputs["angle_p_fc"]
                angle_y_fc = infer_requests_pose[previous_inference_pose].outputs["angle_y_fc"]

                # Parse head pose detection results
                if ((angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) &
                        (angle_p_fc < 22.5)):
                    looking += 1
                    POSE_CHECKED = True
                    INFO = INFO._replace(looker=looking)
                else:
                    INFO = INFO._replace(looker=looking)

                current_inference_pose += 1
                if current_inference_pose >= args.number_infer_requests:
                    current_inference_pose = 0

                previous_inference_pose += 1
                if previous_inference_pose >= args.number_infer_requests:
                    previous_inference_pose = 0
        else:
            INFO = INFO._replace(looker=0)

        writeResults(frame_count, args, res, (det_time_fd * 1000), (det_time_hp_pose * 1000), len(faces), looking, result_file)

        previous_inference += 1
        if previous_inference >= args.number_infer_requests:
            previous_inference = 0

        current_inference += 1
        if current_inference >= args.number_infer_requests:
            current_inference = 0

        if frame_count % 10 == 0:
            progressUpdate(progress_file_path, int(time.time() - infer_time_start), frame_count+1, video_len+1)
        frame = next_frame
    if args.output_dir:
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
            f.write(str(round(total_time, 1)) + '\n')
            f.write(str(frame_count) + '\n')

    del infer_network.net_plugin
    del infer_network.ie
    del infer_network
    del infer_network_pose.net_plugin
    del infer_network_pose.ie
    del infer_network_pose
    
    result_file.close()
    cap.release()

if __name__ == '__main__':
    main()
    sys.exit()