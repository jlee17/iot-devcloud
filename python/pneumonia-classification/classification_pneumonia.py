from __future__ import print_function
import sys
import os,glob
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from PIL import Image
from argparse import ArgumentParser

from utils import load_img, img_to_array, resize_image

def read_image(path):
    image_original = load_img(path, color_mode="rgb")
    img= resize_image(image_original, target_size=(224, 224))
    x = img_to_array(img, data_format='channels_first')
    return [x,image_original]

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str, nargs="+")
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=20, type=int)
    parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")
    parser.add_argument("-o", "--output_dir", help="If set, it will write a video here instead of displaying it",
                        default=None, type=str)

    return parser


def main():
    
    job_id = os.environ['PBS_JOBID']
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    device=args.device

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=device)

    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    net.batch_size = 1

    exec_net = plugin.load(network=net)

    n,c,h,w=net.inputs[input_blob].shape
    files=glob.glob(os.getcwd()+args.input[0])
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    f=open(os.path.join(args.output_dir, 'result'+job_id+'.txt'), 'w')
    f1=open(os.path.join(args.output_dir, 'stats'+job_id+'.txt'), 'w') 
    time_images=[]
    for file in files:
        [image1,image]= read_image(file)
        t0 = time()
        for i in range(args.number_iter):
            res = exec_net.infer(inputs={input_blob: image1})
            #infer_time.append((time()-t0)*1000)
        infer_time = (time() - t0)*1000
        log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
        if args.perf_counts:
            perf_counts = exec_net.requests[0].get_perf_counts()
            log.info("Performance counters:")
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
            for layer, stats in perf_counts.items():
                print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                                  stats['status'], stats['real_time']))
        res_pb = res[out_blob]
        probs=res_pb[0][0]
        print("Probability of having disease= "+str(probs)+", performed in " + str(np.average(np.asarray(infer_time))) +" ms")

        avg_time = round((infer_time/args.number_iter), 1)
        
                    #f.write(res + "\n Inference performed in " + str(np.average(np.asarray(infer_time))) + "ms") 
        f.write("Pneumonia probability: "+ str(probs) + "\n Inference performed in " + str(avg_time) + "ms \n") 
        time_images.append(avg_time)
    f1.write(str(np.average(np.asarray(time_images)))+'\n')
    f1.write(str(1))

if __name__ == '__main__':
    sys.exit(main() or 0)
