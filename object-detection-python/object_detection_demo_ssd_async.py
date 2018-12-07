#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

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


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input",
                        help="Path to video file or image. 'cam' for capturing video stream from camera", required=True,
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("-pt", "--prob_threshold", help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)
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
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)

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
    exec_net = plugin.load(network=net, num_requests=2)
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

    if args.output_dir:
        out_path = os.path.join(args.output_dir, out_file_name+'.mp4')

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    cap = cv2.VideoCapture(input_stream)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur_request_id = 0
    next_request_id = 1
  
    log.info("Starting inference in async mode...")
    log.info("To switch between sync and async modes press Tab button")
    log.info("To stop the sample execution press Esc button")
    job_id = os.environ['PBS_JOBID']
    result_file = open(os.path.join(args.output_dir,'output_'+str(job_id)+'.txt'), "w")
    progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')

    is_async_mode = True
    render_time = 0
    fps_sum = 0
    frame_count = 0
    inf_list = []
    res_list = []
    res_arr = np.zeros((3000, 1, 1, 100, 7))
    try:
        frame_time_start = time.time()
        while cap.isOpened():
            read_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            initial_w = cap.get(3)
            initial_h = cap.get(4)

            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            # Main sync point:
            # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            # in the regular mode we start the CURRENT request and immediately wait for it's completion
            inf_start = time.time()
            if is_async_mode:
                exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
            else:
                exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})


            if exec_net.requests[cur_request_id].wait(-1) == 0:
                inf_end = time.time()
                det_time = inf_end - inf_start
                #Parse detection results of the current request
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                processBoxes(frame_count, res, labels_map, args.prob_threshold, frame, initial_w, initial_h, result_file, det_time)
    
            #
            frame_count+=1
            if frame_count%10 == 0: 
                progress_file = open(progress_file_path, "w")
                progress_file.write(str(round(100*(frame_count/video_len)))+'\n')
                remaining_time = str(round(((time.time()-frame_time_start)/frame_count)*(video_len-frame_count)))
                estimated_time = str(round(((time.time()-frame_time_start)/frame_count)*video_len))
                progress_file.write(remaining_time+'\n')
                progress_file.write(estimated_time+'\n')
                progress_file.flush()
                progress_file.close()
            key = cv2.waitKey(1)
            if key == 27:
                break
            if (9 == key):
                is_async_mode = not is_async_mode
                log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id

 	##End while loop /
        cap.release()
        result_file.close()

        if args.output_dir is None:
            cv2.destroyAllWindows()
        else:
            total_time = time.time() - frame_time_start
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(str(total_time)+'\n')
                f.write(str(frame_count))

    finally:
        del exec_net
        del plugin

if __name__ == '__main__':
    sys.exit(main() or 0)
