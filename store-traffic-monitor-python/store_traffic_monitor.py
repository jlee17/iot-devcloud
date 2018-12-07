#!/usr/bin/env python3
"""
 Author: Andrei Hutuca <andrei.hutuca@gmail.com>
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

##########################################################
# INCLUDES
##########################################################

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy
import time
import datetime
import collections
import threading
import queue
import subprocess

from openvino.inference_engine import IENetwork, IEPlugin

##########################################################
# CONSTANTS
##########################################################

CONF_VIDEODIR = './UI/resources/video_frames/'
CONF_DATAJSON_FILE = './UI/resources/video_data/data.json'
CONF_VIDEOJSON_FILE = './UI/resources/video_data/videolist.json'
CPU_EXTENSION = '/data/reference-sample-data/extension/libcpu_extension.so'
STATS_WINDOW_NAME = 'Statistics'
CAM_WINDOW_NAME_TEMPLATE = 'Video {}'
PROB_THRESHOLD = 0.145
WINDOW_COLUMNS = 5
QUEUE_SIZE = 100
LOOP_VIDEO = False
UI_OUTPUT = False

##########################################################
# GLOBALS
##########################################################

model_xml = ''
model_bin = ''
labels_file = ''
videoCaps = []
display_lock = threading.Lock()
log_lock = threading.Lock()
rolling_log = None
rolling_log_changed = True
frames = 0
frameNames = []

##########################################################
# CLASSES
##########################################################
class FrameInfo:
    def __init__(self, frameNo=None, count=None, timestamp=None):
        self.frameNo = frameNo
        self.count = count
        self.timestamp = timestamp

class VideoCap:
    def __init__(self, cap, req_label, cap_name, is_cam):
        self.cap = cap
        self.req_label = req_label
        self.cap_name = cap_name
        self.is_cam = is_cam
        self.cur_frame = numpy.array([], dtype='uint8')
        self.initial_w = 0
        self.initial_h = 0
        self.frames = 0
        self.cur_frame_count = 0
        self.total_count = 0
        self.last_correct_count = 0
        self.candidate_count = 0
        self.candidate_confidence = 0
        self.closed = False
        self.frame_queue = queue.Queue(QUEUE_SIZE)
        self.countAtFrame = []

        if not is_cam:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.fps = 0

class VideoWriterThread(threading.Thread):
    def __init__(self, videoCap, output_dir, group=None,
                 args=(), kwargs=None, verbose=None):
        super(VideoWriterThread,self).__init__()
        self.output_dir=output_dir
        self.videoCap = videoCap
        self.daemon = True
        self.name = videoCap.cap_name
        self.queue = videoCap.frame_queue
        self.fps = videoCap.fps
        self.v_temp_file = os.path.join(output_dir, 'inference_output_'+self.name.strip().replace(' ','_')+'_temp.mp4')
        self.v_file = os.path.join(output_dir, 'inference_output_'+self.name.strip().replace(' ','_')+'.mp4')
        self.vw = None

        return

    def run(self):
        global rolling_log, rolling_log_changed

        while True:
            # If video capture is not from a cam
            if self.fps:
                start = time.time()
            frame, new_logs = self.queue.get()
            # Quit if empty frame was sent to us.
            if frame is None:
                break
            # Display the current video frame
#            print("{0} : shape {1}".format(self.name, frame.shape))
            if self.vw == None:
                height = numpy.size(frame, 0)
                width = numpy.size(frame, 1)
                self.vw = cv2.VideoWriter(self.v_temp_file, 0x00000021, 15.0, (width, height), True)
            self.vw.write(frame)
            # If available, add the new logs to the rolling log
            if new_logs:
                log_lock.acquire()
                for log in new_logs:
                    rolling_log.append(log)
                rolling_log_changed = True
                log_lock.release()
    
            if self.fps:
                stop = time.time()
                # Limit the frame rate at the video FPS
                time_to_sleep = 1 / self.fps - (stop - start)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
        if self.vw:
            print('Releasing '+self.v_file)
            self.vw.release()
        return

def adjustSpeeds(vc_threads, stats_thread, new_fps):
    #ffmpeg -i input.mkv -filter:v "setpts=0.5*PTS" output.mkv
    pts='setpts={}*PTS'.format(15./new_fps)
    for vc in vc_threads:
        cmd = ['ffmpeg', '-y', '-i', vc.v_temp_file, '-filter:v', pts, vc.v_file]
        subprocess.call(cmd)
        if os.path.exists(vc.v_temp_file):
            os.remove(vc.v_temp_file)
    cmd = ['ffmpeg', '-y', '-i', stats_thread.v_temp_file, '-filter:v', pts, stats_thread.v_file]
    subprocess.call(cmd)
    if os.path.exists(stats_thread.v_temp_file):
        os.remove(stats_thread.v_temp_file)
    
    
class DisplayThread(threading.Thread):
    def __init__(self, videoCap, group=None, target=None,
                 args=(), kwargs=None, verbose=None):
        super(DisplayThread,self).__init__()
        self.videoCap = videoCap
        self.target = target
        self.daemon = True
        self.name = videoCap.cap_name
        self.queue = videoCap.frame_queue
        self.fps = videoCap.fps

        return

    def run(self):
        global rolling_log, rolling_log_changed

        while True:
            # If video capture is not from a cam
            if self.fps:
                start = time.time()
            frame, new_logs = self.queue.get()
            # Display the current video frame
            display_lock.acquire()
            if UI_OUTPUT:
                imgName = self.name
                imgName = imgName.split()[0] + "_" + chr(ord(imgName.split()[1]) + 1)
                imgName += "_" + str(self.videoCap.frames)
                frameNames.append(imgName)
                imgName = CONF_VIDEODIR + imgName + ".jpg"
                cv2.imwrite(imgName, frame)
                a = saveJSON()
                if(a):
                    return a
            else:
                cv2.imshow(self.name, frame)
            display_lock.release()
            # If available, add the new logs to the rolling log
            if new_logs:
                log_lock.acquire()
                for log in new_logs:
                    rolling_log.append(log)
                rolling_log_changed = True
                log_lock.release()

            if self.fps:
                stop = time.time()
                # Limit the frame rate at the video FPS
                time_to_sleep = 1 / self.fps - (stop - start)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        return

##########################################################
# FUNCTIONS
##########################################################

def parse_conf_file(conf_file):
    """
        Parses the configuration file.
        Reads model_xml, model_bin, labels_file, videoCaps
    """
    global model_xml, model_bin, labels_file, videoCaps

    with open(conf_file, 'r') as f:
        model_xml = f.readline().rstrip('\n')
        model_bin = f.readline().rstrip('\n')
        labels_file = f.readline().rstrip('\n')

        for idx, item in enumerate(f.read().splitlines()):
            split = item.split()
            if split[0].isdigit():
                videoCap = VideoCap(cv2.VideoCapture(int(split[0])), split[1], CAM_WINDOW_NAME_TEMPLATE.format(idx), True)
            else:
                videoCap = VideoCap(cv2.VideoCapture(split[0]), split[1], CAM_WINDOW_NAME_TEMPLATE.format(idx), False)
            videoCaps.append(videoCap)

def arrange_windows(width, height, output_dir):
    """
        Arranges the windows so they are not overlapping
        Also starts the display threads
    """
    spacer = 25
    cols = 0
    rows = 0

    # Arrange video windows
    for idx in range(len(videoCaps)):
        if(cols == WINDOW_COLUMNS):
            cols = 0
            rows += 1
        cv2.namedWindow(CAM_WINDOW_NAME_TEMPLATE.format(idx), cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(CAM_WINDOW_NAME_TEMPLATE.format(idx), (spacer + width) * cols, (spacer + height) * rows)
        cols += 1
        DisplayThread(videoCaps[idx], output_dir).start()

    # Arrange statistics window
    if(cols == WINDOW_COLUMNS):
        cols = 0
        rows += 1
    cv2.namedWindow(STATS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(STATS_WINDOW_NAME, (spacer + width) * cols, (spacer + height) * rows)

def saveJSON():
    """
        This JSON contains info about current and total object count
    """
    if UI_OUTPUT:
        dataJSON = open(CONF_DATAJSON_FILE, "w")
        if not os.access(CONF_DATAJSON_FILE, os.W_OK):
            print("Could not open dataJSON file for writing")
            return 5

        videoJSON = open(CONF_VIDEOJSON_FILE, "w")
        if not os.access(CONF_VIDEOJSON_FILE, os.W_OK):
            print("Could not open videoJSON file for writing")
            return 5

        dataJSON.write("{\n");
        videoJSON.write("{\n");
        vsz = len(videoCaps)

        for i in range(vsz):
            if len(videoCaps[i].countAtFrame) > 0:
                j = 0
                dataJSON.write("\t\"Video_" + str(i + 1) + "\": {\n")
                fsz = len(videoCaps[i].countAtFrame) - 1
                for j in range(fsz):
                    strt = "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"time\":\"%s\"\n\t\t},\n" % \
                                (videoCaps[i].countAtFrame[j].frameNo, videoCaps[i].countAtFrame[j].count, videoCaps[i].countAtFrame[j].timestamp)
                    dataJSON.write(strt)

                strt = "\t\t\"%d\": {\n\t\t\t\"count\":\"%d\",\n\t\t\t\"time\":\"%s\"\n\t\t}\n" % \
                    (videoCaps[i].countAtFrame[j].frameNo, videoCaps[i].countAtFrame[j].count, videoCaps[i].countAtFrame[j].timestamp)
                dataJSON.write(strt)
                dataJSON.write("\t},\n")

        dataJSON.write("\t\"totals\": {\n")
        for i in range(vsz - 1):
            dataJSON.write("\t\t\"Video_" + str(i + 1) + "\": \"" + str(videoCaps[i].total_count) + "\",\n")

        i = vsz - 1
        dataJSON.write("\t\t\"Video_" + str(i + 1) + "\": \"" + str(videoCaps[i].total_count) + "\"\n")
        dataJSON.write("\t}\n")
        dataJSON.write("}")
        dataJSON.close()

        sz = len(frameNames) - 1
        for i in range(sz):
            videoJSON.write("\t\"" + str(i + 1) + "\":\"" + str(frameNames[i]) + "\",\n")

        i = sz
        videoJSON.write("\t\"" + str(i + 1) + "\":\"" + str(frameNames[i]) + "\"\n")
        videoJSON.write("}")
        videoJSON.close()

        return 0

def main(device, output_dir, conf):
    global rolling_log, rolling_log_changed

    parse_conf_file(conf)
    # Plugin initialization for specified device and load extensions library
    print("Initializing plugin for {} device...".format(device))
    plugin = IEPlugin(device=device)
    if 'CPU' in device:
        print("Loading CPU extensions")
        plugin.add_cpu_extension(CPU_EXTENSION)

    # Read IR
    print("Reading IR...")
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Load the IR
    print("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net

    # Read the labels file
    if labels_file:
        with open(labels_file, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # Init a rolling log to store events
    rolling_log_size = int((h - 15) / 20)
    rolling_log = collections.deque(maxlen=rolling_log_size)

    # Init inference request IDs
    cur_request_id = 0
    next_request_id = 1
    # Start with async mode enabled
    is_async_mode = True

    stats_video=None
    stats_thread=None
    vc_threads=[]
    if output_dir:
        stats_video = VideoCap(None, 'STATS', 'stats', True)
        stats_thread = VideoWriterThread(stats_video,output_dir)
        stats_thread.start()
        for vc in videoCaps:
            new_vc_thread = VideoWriterThread(vc, output_dir)
            new_vc_thread.start()
            vc_threads.append(new_vc_thread)
    elif UI_OUTPUT:
        for idx in range(len(videoCaps)):
            DisplayThread(videoCaps[idx]).start()
    else:
        # Arrange windows so they are not overlapping
        arrange_windows(w, h)

    if output_dir == None:
        print("To stop the execution press Esc button")
    else:
        print("Wrting results into {}".format(output_dir))
    total_frames_count=0
    fps_sum=0.
    while True:
        # If all video captures are closed stop the loop
        if False not in [videoCap.closed for videoCap in videoCaps]:
            break
        total_frames_count+=1

        iter_frames_processed=0
        t_start = time.time()
        # loop over all video captures
        for idx, videoCapInfer in enumerate(videoCaps):
            # read the next frame
            ret, frame = videoCapInfer.cap.read()
            # If the read failed
            #  close the window
            #  mark it as closed
            #  go to the next video capture
            if not ret:
                if not videoCapInfer.closed and output_dir ==None:
                    cv2.destroyWindow(videoCapInfer.cap_name)
                videoCapInfer.closed = True
                continue

            # Copy the current frame for later use
            videoCapInfer.cur_frame = frame.copy()
            videoCapInfer.initial_w = videoCapInfer.cap.get(3)
            videoCapInfer.initial_h = videoCapInfer.cap.get(4)
            # Resize and change the data layout so it is compatible
            in_frame = cv2.resize(videoCapInfer.cur_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))

            if is_async_mode:
                exec_net.start_async(request_id=next_request_id, inputs={input_blob: in_frame})
                # Async enabled and only one video capture
                if(len(videoCaps) == 1):
                    videoCapResult = videoCapInfer
                # Async enabled and more than one video capture
                else:
                    # Get previous index
                    videoCapResult = videoCaps[idx - 1 if idx - 1 >= 0 else len(videoCaps) - 1]
            else:
                # Async disabled
                exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
                videoCapResult = videoCapInfer

            if exec_net.requests[cur_request_id].wait(-1) == 0:
                current_count = 0;
                # Parse detection results of the current request
                res = exec_net.requests[cur_request_id].outputs[out_blob]
                for obj in res[0][0]:
                    class_id = int(obj[1])
                    # Draw only objects when probability more than specified threshold
                    if (obj[2] > PROB_THRESHOLD and
                        videoCapResult.req_label in labels_map and
                        labels_map.index(videoCapResult.req_label) == class_id - 1):
                        current_count += 1
                        xmin = int(obj[3] * videoCapResult.initial_w)
                        ymin = int(obj[4] * videoCapResult.initial_h)
                        xmax = int(obj[5] * videoCapResult.initial_w)
                        ymax = int(obj[6] * videoCapResult.initial_h)
                        # Draw box
                        cv2.rectangle(videoCapResult.cur_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4, 16)

                if videoCapResult.candidate_count is current_count:
                    videoCapResult.candidate_confidence += 1
                else:
                    videoCapResult.candidate_confidence = 0
                    videoCapResult.candidate_count = current_count

                new_logs = []
                if videoCapResult.candidate_confidence is 5:
                    videoCapResult.candidate_confidence = 0
                    if current_count > videoCapResult.last_correct_count:
                        videoCapResult.total_count += current_count - videoCapResult.last_correct_count

                    if current_count is not videoCapResult.last_correct_count:
                        if UI_OUTPUT:
                            currtime = datetime.datetime.now().strftime("%H:%M:%S")
                            fr = FrameInfo(videoCapResult.frames, current_count, currtime)
                            videoCapResult.countAtFrame.append(fr)
                        new_objects = current_count - videoCapResult.last_correct_count
                        for _ in range(new_objects):
                            strt = "{} - {} detected on {}".format(time.strftime("%H:%M:%S"), videoCapResult.req_label, videoCapResult.cap_name)
                            new_logs.append(strt)

                    videoCapResult.frames+=1
                    videoCapResult.last_correct_count = current_count
                else:
                    videoCapResult.frames+=1

                videoCapResult.cur_frame = cv2.resize(videoCapResult.cur_frame, (w, h))

                if not UI_OUTPUT:
                    # Add log text to each frame
                    log_message = "Async mode is on." if is_async_mode else \
                                  "Async mode is off."
                    cv2.putText(videoCapResult.cur_frame, log_message, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    log_message = "Total {} count: {}".format(videoCapResult.req_label, videoCapResult.total_count)
                    cv2.putText(videoCapResult.cur_frame, log_message, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    log_message = "Current {} count: {}".format(videoCapResult.req_label, videoCapResult.last_correct_count)
                    cv2.putText(videoCapResult.cur_frame, log_message, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Send the current video frame and the new logs to the propper display thread
                videoCapResult.frame_queue.put((videoCapResult.cur_frame, new_logs))

                if not UI_OUTPUT:
                    # Display stats
                    log_lock.acquire()
                    if rolling_log_changed:
                        if output_dir == None:
                            stats = numpy.zeros((h if h > (len(videoCaps) * 20 + 15) else (len(videoCaps) * 20 + 15), w if w > 345 else 345, 1), dtype = 'uint8')
                            for i, log in enumerate(rolling_log):
                                cv2.putText(stats, log, (10, i * 20 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            display_lock.acquire()
                            cv2.imshow(STATS_WINDOW_NAME, stats)
                            display_lock.release()
                        rolling_log_changed = False
                    log_lock.release()

            # imshow and waitKey are not thread safe
            display_lock.acquire()
            key = cv2.waitKey(1)
            display_lock.release()
            # Esc key pressed
            if key == 27:
                cv2.destroyAllWindows()
                del exec_net
                del plugin
                print("Finished")
                return
            # Tab key pressed
            if key == 9:
                is_async_mode = not is_async_mode
                print("Switched to {} mode".format("async" if is_async_mode else "sync"))

            if is_async_mode:
                # Swap infer request IDs
                cur_request_id, next_request_id = next_request_id, cur_request_id

            if LOOP_VIDEO and videoCapInfer.is_cam:
                videoCapInfer.cur_frame_count += 1
                # If a video capture has ended restart it
                if (videoCapInfer.cur_frame_count == videoCapInfer.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    videoCapInfer.cur_frame_count = 0
                    videoCapInfer.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            iter_frames_processed+=1

        t_finished = time.time()
        #iter_fps=float(iter_frames_processed)/(t_finished-t_start)
        iter_fps=1.0/(t_finished-t_start)
        fps_sum += iter_fps

        if output_dir:
            stats = numpy.zeros((h, w, 3), dtype = 'uint8')
            for i, log in enumerate(rolling_log):
                cv2.putText(stats, log, (10, i * 20 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            stats_video.frame_queue.put((stats, None))



    # Teminating video writes
    if output_dir:
        stats_video.frame_queue.put((None, None))
        for vc in videoCaps:
            vc.frame_queue.put((None, None))
        stats_thread.join()
        for vc_thread in vc_threads:
            vc_thread.join()
        fps=fps_sum/float(total_frames_count)
        print('total_fps was {}'.format(fps))
        adjustSpeeds(vc_threads, stats_thread, fps)
        with open(os.path.join(output_dir, 'fps.txt'), 'w') as f:
            f.write(str(fps))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output_dir', default=None, help="Name of the directory to write the video to")
    parser.add_argument('-d', '--device', default='GPU', help="Device to use for inference")
    parser.add_argument('-c', '--conf', default='./resources/conf.txt', help=".conf file to use")

    args = vars(parser.parse_args())
    allowed_devices = ['GPU', 'CPU', 'MYRIAD']
    if not args['device'] in allowed_devices:
        print("Device must be one of {}".format(allowed_devices))
        sys.exit(1)
    sys.exit(main(**args) or 0)
