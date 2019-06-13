#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IENetwork, IEPlugin
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
from demoTools.demoutils import progressUpdate
import json

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Path to an .xml file with a trained model.',
                        required=True,
                        type=str)
    parser.add_argument('-i', '--input',
                        help='Path to video file or image. \'cam\' for capturing video stream from camera.',
                        required=True,
                        type=str)
    parser.add_argument('-ce', '--cpu_extension',
                        help='MKLDNN-targeted custom layers.'
                             'Absolute path to a shared library with the kernel implementation.',
                        type=str,
                        default=None)
    parser.add_argument('-pp', '--plugin_dir',
                        help='Path to a plugin directory.',
                        type=str,
                        default=None)
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on; CPU, GPU, FPGA, MYRIAD, or HDDL is acceptable.'
                             'Demo will look for a suitable plugin for specified device (CPU by default).',
                        default='CPU',
                        type=str)
    parser.add_argument('-nireq', '--number_infer_requests',
                        help='Number of parallel inference requests (default is 2).',
                        type=int,
                        required=False,
                        default=2)
    parser.add_argument('-s', '--show',
                        help='Show preview to the user.',
                        action='store_true',
                        required=False)
    parser.add_argument('-l', '--labels',
                        help='Labels mapping file.',
                        default=None,
                        type=str)
    parser.add_argument('-pt', '--prob_threshold',
                        help='Probability threshold for detection filtering.',
                        default=0.5,
                        type=float)
    parser.add_argument('-o', '--output_dir',
                        help='Location to store the results of the processing',
                        default=None,
                        required=True,
                        type=str)
    return parser


def processBoxes(frame_count, res, labels_map, prob_threshold, frame, initial_w, initial_h, result_file, det_time):
    for obj in res[0][0]:
        dims = ""
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            xmin = str(int(obj[3] * initial_w))
            ymin = str(int(obj[4] * initial_h))
            xmax = str(int(obj[5] * initial_w))
            ymax = str(int(obj[6] * initial_h))
            class_id = str(int(obj[1]))
            est = str(round(obj[2]*100, 1))
            time = round(det_time*1000)
            out_list = [str(frame_count), xmin, ymin, xmax, ymax, class_id, est, str(time)]
            for i in range(len(out_list)):
                dims += out_list[i]+' '
            dims += '\n'
            result_file.write(dims)


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Initializing plugin for {} device...".format(args.device))
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading plugins for {} device...".format(args.device))
        plugin.add_cpu_extension(args.cpu_extension)

    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=args.number_infer_requests)
 
    # Read and pre-process input image
    if isinstance(net.inputs[input_blob], list):
        n, c, h, w = net.inputs[input_blob]
    else:
        n, c, h, w = net.inputs[input_blob].shape
    del net

    if args.input == 'cam':
        input_stream = 0
        out_file_name = 'cam'
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        out_file_name = os.path.splitext(os.path.basename(args.input))[0]

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_inference = 0
    required_inference_requests_were_executed = False
    previous_inference = 1 - args.number_infer_requests
    step = 0
    steps_count = args.number_infer_requests - 1

    infer_requests = exec_net.requests
  
    log.info("Starting inference in async mode, {} requests in parallel...".format(args.number_infer_requests))
    result_file = open(os.path.join(args.output_dir, 'output.txt'), "w")
    progress_file_path = os.path.join(args.output_dir, 'i_progress.txt')

    frame_count = 0
    frames = []
    try:
        infer_time_start = time.time()
        while not required_inference_requests_were_executed or step < steps_count or cap.isOpened():
            read_time = time.time()
            ret, next_frame = cap.read()
            frames.append(next_frame)
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)

            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            
            # In the truly Async mode, we start the NEXT infer request, while waiting for the CURRENT to complete
            inf_start = time.time()
            exec_net.start_async(request_id=current_inference, inputs={input_blob: in_frame})
            
            if previous_inference >= 0:
                status = infer_requests[previous_inference].wait()
                if status is not 0:
                    raise Exception("Infer request not completed successfully")

                inf_end = time.time()
                det_time = inf_end - inf_start

                # Parse detection results of the current request
                res = infer_requests[previous_inference].outputs[out_blob]
                frame = frames.pop(0)
                processBoxes(frame_count, res, labels_map, args.prob_threshold, frame,
                             initial_w, initial_h, result_file, det_time)

                frame_count += 1

                # Write data to progress tracker
                if frame_count%10 == 0: 
                    progressUpdate(progress_file_path, time.time()-infer_time_start, frame_count, video_len) 

            current_inference += 1
            if current_inference >= args.number_infer_requests:
                current_inference = 0
                required_inference_requests_were_executed = True

            previous_inference += 1
            if previous_inference >= args.number_infer_requests:
                previous_inference = 0

            step += 1
        # End while loop

        frame_count += (args.number_infer_requests - 1)

        stats = {}
        stats['time'] = round(time.time() - infer_time_start, 2)
        stats['frame'] = frame_count
        stats['fps'] = round(frame_count/(time.time() - infer_time_start), 2)
        stats_file = args.output_dir + '/stats.json'

        with open(stats_file, 'w') as f:
            json.dump(stats, f)

        cap.release()
        result_file.close()

        progressUpdate(progress_file_path, time.time()-infer_time_start, frame_count, video_len)

    finally:
        log.info("Processing done...")
        del exec_net
        del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
