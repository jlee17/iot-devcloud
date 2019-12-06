"""People Counter."""
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
import time
import socket
import json
import cv2

import logging as log

from argparse import ArgumentParser
from inference import Network
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import *

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument('-nireq', '--number_infer_requests',
                        help='Number of parallel inference requests (default is 2).',
                        type=int,
                        required=False,
                        default=2)
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)
    return parser


def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


def ssd_out(frame_count,result,result_file):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
            dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {est} {time} {current_count} \n".format(frame_id=frame_count, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, class_id=int(obj[1]), est=round(obj[2]*100, 1), time='N/A', current_count=current_count)
            result_file.write(dims)
    return current_count


def main():
    """
    Load the network and parse the SSD output.
    :return: None
    """

    args = build_argparser().parse_args()

    # Flag for the input image
    single_image_mode = False
    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    start_time = 0
    result=None
    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          args.number_infer_requests, args.cpu_extension)
    # Checks for live feed
    #if args.input == 'CAM':
        #input_stream = 0

    # Checks for input image
    if args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream) 
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    job_id = os.environ['PBS_JOBID']
    #job_id = "12345"
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')
    infer_time_start = time.time()
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    #people_counter = cv2.VideoWriter(os.path.join(args.output_dir, "people_counter.mp4"), cv2.VideoWriter_fourcc(*"AVC1"), fps, (int(initial_w), int(initial_h)), True)
    result_file = open(os.path.join(args.output_dir, 'output_'+job_id+'.txt'), "w")
    while frame_count<video_len:
        flag, frame = cap.read()
        if flag:
            # Start async inference
            image = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            image = image.transpose((2, 0, 1))
            image = image.reshape((n, c, h, w))
            # Start asynchronous inference for specified request.
            inf_start = time.time()
            infer_network.exec_net(current_inference, image)
        if previous_inference >= 0:
            # Wait for the result
            status=infer_network.wait(previous_inference)
            #if status is not 0:
                #raise Exception("Infer request not completed successfully")
            # Results of the output layer of the network
            result = infer_network.get_output(previous_inference)
            if args.perf_counts:
                perf_count = infer_network.performance_counter(previous_inference)
                performance_counts(perf_count)
            current_count = ssd_out(frame_count,result,result_file)
            frame_count += 1
        if frame_count%10 == 0 or frame_count >= video_len: 
            progressUpdate(progress_file_path, int(time.time()-infer_time_start), frame_count+1, video_len+1)

        # Increment counter for the inference queue and roll them over if necessary 
        current_inference += 1
        if current_inference >= args.number_infer_requests:
            current_inference = 0
        previous_inference += 1
        if previous_inference >= args.number_infer_requests:
            previous_inference = 0
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    if args.output_dir:
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
            f.write(str(round(total_time, 1))+'\n')
            f.write(str(frame_count)+'\n')
    result_file.close()
    cap.release()
    infer_network.clean()


if __name__ == '__main__':
    main()
    exit(0)