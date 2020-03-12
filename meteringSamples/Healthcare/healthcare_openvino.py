import os
import sys
import psutil
import logging as log
import numpy as np
import keras as K
import h5py
import time 
import tensorflow as tf 
from openvino.inference_engine import IENetwork, IECore
from distutils.sysconfig import get_python_lib
packages_directory=get_python_lib()
import matplotlib.pyplot as plt
from qarpo.demoutils import progressUpdate
from argparser import args
import applicationMetricWriter

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K

import matplotlib.pyplot as plt

onnx=False
#TODO - Enable nGraph Bridge - Switch to (decathlon) venv!

if onnx:
    #TODO - Include ngraph onnx backend
    import onnx
    from ngraph_onnx.onnx_importer.importer import import_onnx_model
    import ngraph as ng

print ("We are using Tensorflow version", tf.__version__,\
       "with Intel(R) MKL", "enabled" if tf.pywrap_tensorflow.IsMklEnabled() else "disabled",)

def print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob, args):
    """
    Prints layer by layer inference times.
    Good for profiling which ops are most costly in your model.
    """

    # Start sync inference
    print("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []

    for i in range(args.number_iter):
        input_data_transposed_1=input_data[0:batch_size].transpose(0,3,1,2)
        t0 = time.time()
        res = exec_net.infer(inputs={input_blob: input_data_transposed_1[:,:n_channels]})
        infer_time.append((time.time() - t0) * 1000)


    average_inference = np.average(np.asarray(infer_time))
    print("Average running time of one batch: {:.5f} ms".format(average_inference))
    print("Images per second = {:.3f}".format(batch_size * 1000.0 / average_inference))

    perf_counts = exec_net.requests[0].get_perf_counts()
    log.info("Performance counters:")
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format("name",
                                                         "layer_type",
                                                         "exec_type",
                                                         "status",
                                                         "real_time, us"))
    for layer, stats in perf_counts.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                             stats["layer_type"],
                                                             stats["exec_type"],
                                                             stats["status"],
                                                             stats["real_time"]))


def plot_predictions(predictions, input_data, label_data, img_indicies, args):
    """
    Plot the predictions with matplotlib and save to png files
    """
    png_directory = "inference_examples_openvino"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    import matplotlib.pyplot as plt

    # Processing output blob
    print("Plotting the predictions and saving to png files. Please wait...")
    number_imgs = predictions.shape[0]
    num_rows_per_image = args.rows_per_image
    row = 0

    for idx in range(number_imgs):

        if row==0:  plt.figure(figsize=(15,15))

        plt.subplot(num_rows_per_image, 3, 1+row*3)
        plt.imshow(input_data[idx,0,:,:], cmap="bone", origin="lower")
        plt.axis("off")
        if row==0: plt.title("MRI")

        plt.subplot(num_rows_per_image, 3, 2+row*3)
        plt.imshow(label_data[idx,0,:,:], origin="lower")
        plt.axis("off")
        if row==0: plt.title("Ground truth")

        plt.subplot(num_rows_per_image, 3, 3+row*3)
        plt.imshow(predictions[idx,0,:,:], origin="lower")
        plt.axis("off")
        if row ==0:  plt.title("Prediction")

        plt.tight_layout()

        if (row == (num_rows_per_image-1)) or (idx == (number_imgs-1)):

            if num_rows_per_image==1:
                fileidx = "pred{}.png".format(img_indicies[idx])
            else:
                fileidx = "pred_group{}".format(idx // num_rows_per_image)
            filename = os.path.join(png_directory, fileidx)
            plt.savefig(filename,
                        bbox_inches="tight", pad_inches=0)
            print("Saved file: {}".format(filename))
            row = 0
        else:
            row += 1



def load_data():
    """
    Modify this to load your data and labels
    """

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    df = h5py.File(data_fn, "r")
    imgs_validation = df["imgs_validation"]
    msks_validation = df["msks_validation"]
    img_indicies = range(len(imgs_validation))

    """
    OpenVINO uses channels first tensors (NCHW).
    TensorFlow usually does channels last (NHWC).
    So we need to transpose the axes.
    """
    input_data = imgs_validation
    msks_data = msks_validation
    return input_data, msks_data, img_indicies


def load_model( ):
    """
    Load the OpenVINO model.
    """
    log.info("Loading U-Net model to the plugin")
    model_xml = args.intermediate_rep +".xml"
    model_bin = args.intermediate_rep +".bin"
    return model_xml, model_bin

def calc_dice(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def dice_coef(y_true, y_pred, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2), smooth=1.):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.log(2.*numerator) + tf.log(denominator)

    return dice_loss


def combined_dice_ce_loss(y_true, y_pred, axis=(1, 2), smooth=1.,
                          weight=0.9):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    return weight*dice_coef_loss(y_true, y_pred, axis, smooth) + \
        (1-weight)*K.losses.binary_crossentropy(y_true, y_pred)

def plotDiceScore(img_no,img,msk,pred_mask,plot_result, time):
    dice_score = calc_dice(pred_mask, msk)

    if plot_result:
        plt.figure(figsize=(15, 15))
        plt.suptitle("Time for prediction TF: {} ms".format(time), x=0.1, y=0.70,  fontsize=20, va="bottom")
        plt.subplot(1, 3, 1)
        plt.imshow(img[0,0,:,:], cmap="bone", origin="lower")
        plt.axis("off")
        plt.title("MRI Input", fontsize=20)
        plt.subplot(1, 3, 2)
        plt.imshow(msk[0,0, :, :], origin="lower")
        plt.axis("off")
        plt.title("Ground truth", fontsize=20)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[0,0, :, :], origin="lower")
        plt.axis("off")
        plt.title("Prediction\nDice = {:.4f}".format(dice_score), fontsize=20)

        plt.tight_layout()

        png_name = os.path.join(png_directory, "pred{}.png".format(img_no))
        plt.savefig(png_name, bbox_inches="tight", pad_inches=0)


# Create output directory for images
job_id = os.environ['PBS_JOBID']
png_directory = os.path.join(args.results_directory, job_id)
if not os.path.exists(png_directory):
    os.makedirs(png_directory)
 

data_fn = os.path.join(args.data_path, args.data_filename)
model_fn = os.path.join(args.output_path, args.inference_filename)



log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

# Plugin initialization for specified device and load extensions library if specified
print("check")
#plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
ie=IECore()
print(args.device)
if args.cpu_extension and "CPU" in args.device:
    #plugin.add_cpu_extension(args.cpu_extension)
    ie.add_extension(args.cpu_extension, "CPU")

# Read IR
# If using MYRIAD then we need to load FP16 model version
model_xml, model_bin = load_model()
log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = IENetwork(model=model_xml, weights=model_bin)
if args.device == "CPU":
        supported_layers = ie.query_network(net, args.device)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
assert len(net.outputs) == 1, "Sample supports only single output topologies"


"""
Ask OpenVINO for input and output tensor names and sizes
"""
input_blob = next(iter(net.inputs))  # Name of the input layer
out_blob = next(iter(net.outputs))   # Name of the output layer

batch_size, n_channels, height, width = net.inputs[input_blob].shape
batch_size, n_out_channels, height_out, width_out = net.outputs[out_blob].shape
net.batch_size = batch_size

# Load data
input_data, label_data, img_indicies = load_data()



# Loading model to the plugin
exec_net = ie.load_network(network=net,device_name=args.device)
# del net

args.stats = True
if args.stats:
    # Print the latency and throughput for inference
    print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob, args)

indicies_validation = [40, 63, 43, 55, 99, 101, 19, 46] #[40]
val_id = 1
infer_time = 0
process_time_start = time.time()
progress_file_path = os.path.join(png_directory, "i_progress.txt")
for idx in indicies_validation:

    input_data_transposed=input_data[idx:(idx+batch_size)].transpose(0,3,1,2)
    start_time = time.time()
    inf_start = time.time()
    res = exec_net.infer(inputs={input_blob:input_data_transposed[:,:n_channels]})
    # Save the predictions to array
    predictions = res[out_blob]
    time_elapsed = time.time()-start_time
    infer_time += time_elapsed
    det_time = time.time() - inf_start
    applicationMetricWriter.send_inference_time(det_time*1000)
    
    plotDiceScore(idx,input_data_transposed,label_data[[idx]].transpose(0,3,1,2),predictions,True, round(time_elapsed*1000))
    progressUpdate(progress_file_path, time.time()-process_time_start, val_id, len(indicies_validation)) 
    val_id += 1


total_time = time.time() - process_time_start
with open(os.path.join(png_directory, 'stats.txt'), 'w') as f:
                f.write(str(round(infer_time, 4))+'\n')
                f.write(str(val_id)+'\n')
                f.write("Frames processed per second = {}".format(round(val_id/infer_time)))
                
applicationMetricWriter.send_application_metrics(model_xml, args.device)

