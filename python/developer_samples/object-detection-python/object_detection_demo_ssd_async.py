


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
# /usr/bin/env python
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import time
import logging as log
import numpy as np
import io
from openvino.inference_engine import IENetwork, IECore
from qarpo.demoutils import *
import cv2



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
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on; CPU, GPU, FPGA, MYRIAD, or HDDL is acceptable.'
                             '(CPU by default).',
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

def processBoxes(frame_count, res, labels_map, prob_threshold, initial_w, initial_h, result_file):
    for obj in res[0][0]:
        dims = ""
       # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
           dims = "{frame_id} {xmin} {ymin} {xmax} {ymax} {class_id} {est} {time} \n".format(frame_id=frame_count, xmin=int(obj[3] * initial_w), ymin=int(obj[4] * initial_h), xmax=int(obj[5] * initial_w), ymax=int(obj[6] * initial_h), class_id=int(obj[1]), est=round(obj[2]*100, 1), time='N/A')
           result_file.write(dims)

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Creating IECore
    log.info("Initializing IECore...")
    ie = IECore()

    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    supported_layers = ie.query_network(net, "CPU")
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the specified device {}:\n {}".
                    format(args.device, ', '.join(not_supported_layers)))
        log.error("If you are using a custom network, you may need OpenVINO extensions in "
                  "order to use your model. See OpenVINO documention for more.)
        sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    if args.input == 'cam':
        input_stream = 0
        out_file_name = 'cam'
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        out_file_name = os.path.splitext(os.path.basename(args.input))[0]

    log.info("Generating ExecutableNetwork...")
    exec_net = ie.load_network(network=net, num_requests=args.number_infer_requests, device_name=args.device)
 

    log.info("Starting inference in async mode, {} requests in parallel...".format(args.number_infer_requests))
    job_id = str(os.environ['PBS_JOBID'])
    result_file = open(os.path.join(args.output_dir, 'output_'+job_id+'.txt'), "w")
    pre_infer_file = os.path.join(args.output_dir, 'pre_progress_'+job_id+'.txt')
    infer_file = os.path.join(args.output_dir, 'i_progress_'+job_id+'.txt')
    processed_vid = '/tmp/processed_vid.bin'


    # Read and pre-process input image
    if isinstance(net.inputs[input_blob], list):
        n, c, h, w = net.inputs[input_blob]
    else:
        n, c, h, w = net.inputs[input_blob].shape
    del net


    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_len < args.number_infer_requests:
        args.number_infer_requests = video_len 
    #Pre inference processing, read mp4 frame by frame, process using openCV and write to binary file
    width = int(cap.get(3))
    height = int(cap.get(4))
    CHUNKSIZE = n*c*w*h
    id_ = 0
    with open(processed_vid, 'w+b') as f:
        time_start = time.time()
        while cap.isOpened():
            ret, next_frame = cap.read()
            if not ret:
                break
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            bin_frame = bytearray(in_frame) 
            f.write(bin_frame)
            id_ += 1
            if id_%10 == 0: 
                progressUpdate(pre_infer_file, time.time()-time_start, id_, video_len) 
    cap.release()

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    
    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    infer_requests = exec_net.requests
    frame_count = 0

    try:
        infer_time_start = time.time()
        with open(processed_vid, "rb") as data:
            while frame_count < video_len:
                # Read next frame from input stream if available and submit it for inference 
                byte = data.read(CHUNKSIZE)
                if not byte == b"":
                    deserialized_bytes = np.frombuffer(byte, dtype=np.uint8)
                    in_frame = np.reshape(deserialized_bytes, newshape=(n, c, h, w))
                    exec_net.start_async(request_id=current_inference, inputs={input_blob: in_frame})
                
                # Retrieve the output of an earlier inference request
                if previous_inference >= 0:
                    status = infer_requests[previous_inference].wait()
                    if status is not 0:
                        raise Exception("Infer request not completed successfully")
                    # Parse inference results
                    res = infer_requests[previous_inference].outputs[out_blob]
                    processBoxes(frame_count, res, labels_map, args.prob_threshold, width, height, result_file)
                    frame_count += 1

                # Write data to progress tracker
                if frame_count % 10 == 0: 
                    progressUpdate(infer_file, time.time()-infer_time_start, frame_count+1, video_len+1) 

                # Increment counter for the inference queue and roll them over if necessary 
                current_inference += 1
                if current_inference >= args.number_infer_requests:
                    current_inference = 0

                previous_inference += 1
                if previous_inference >= args.number_infer_requests:
                    previous_inference = 0

        # End while loop
        total_time = time.time() - infer_time_start
        with open(os.path.join(args.output_dir, 'stats_{}.txt'.format(job_id)), 'w') as f:
                f.write('{:.3g} \n'.format(total_time))
                f.write('{} \n'.format(frame_count))

        result_file.close()

    finally:
        log.info("Processing done...")
        del exec_net


if __name__ == '__main__':
    sys.exit(main() or 0)

