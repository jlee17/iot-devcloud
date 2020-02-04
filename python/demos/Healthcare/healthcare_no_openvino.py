import os
import sys
import psutil
import logging as log
import numpy as np
import keras as K
import h5py
import time 
import tensorflow as tf 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from distutils.sysconfig import get_python_lib
from qarpo import progressUpdate
import json

packages_directory=get_python_lib()
print ("We are using Tensorflow version", tf.__version__,\
       "with Intel(R) MKL", "enabled" if tf.pywrap_tensorflow.IsMklEnabled() else "disabled",)


from argparser import args

if args.keras_api:
    import keras as K
else:
    from tensorflow import keras as K


onnx=False
#TODO - Enable nGraph Bridge - Switch to (decathlon) venv!

if onnx:
    #TODO - Include ngraph onnx backend
    import onnx
    from ngraph_onnx.onnx_importer.importer import import_onnx_model
    import ngraph as ng

print ("We are using Tensorflow version", tf.__version__,\
       "with Intel(R) MKL", "enabled" if tf.pywrap_tensorflow.IsMklEnabled() else "disabled",)

# os.environ['MKLDNN_VERBOSE'] = "2"

# Create output directory for images
job_id = os.environ['PBS_JOBID']
job_id = job_id.rstrip().split('.')[0]
png_directory = os.path.join(args.results_directory, job_id)
if not os.path.exists(png_directory):
    os.makedirs(png_directory)
    
data_fn = os.path.join(args.data_path, args.data_filename)
model_fn = os.path.join(args.output_path, args.inference_filename)
print(data_fn)
print(model_fn)
print("Done")
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


df = h5py.File(data_fn, "r")
imgs_validation = df["imgs_validation"]
msks_validation = df["msks_validation"]


model = K.models.load_model(model_fn, custom_objects={	"combined_dice_ce_loss": combined_dice_ce_loss,
							"dice_coef_loss": dice_coef_loss,
							"dice_coef": dice_coef})

def plotDiceScore(img_no,img,msk,pred_mask,plot_result, time):
    dice_score = calc_dice(pred_mask, msk)

    #print("Dice score for Image #{} = {:.4f}".format(img_no,
    #                                                 dice_score))
    if plot_result:
        plt.figure(figsize=(15, 15))
        plt.suptitle("Time for prediction TF: {} ms".format(time), x=0.1, y=0.70,  fontsize=20, va="bottom")
        plt.subplot(1, 3, 1)
        plt.imshow(img[0, :, :, 0], cmap="bone", origin="lower")
        plt.axis("off")
        plt.title("MRI Input", fontsize=20)
        plt.subplot(1, 3, 2)
        plt.imshow(msk[0, :, :, 0], origin="lower")
        plt.axis("off")
        plt.title("Ground truth", fontsize=20)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[0, :, :, 0], origin="lower")
        plt.axis("off")
        plt.title("Prediction\nDice = {:.4f}".format(dice_score), fontsize=20)


        plt.tight_layout()

        png_name = os.path.join(png_directory, "pred{}.png".format(img_no))
        plt.savefig(png_name, bbox_inches="tight", pad_inches=0)
        #print("Saved png file to {}".format(png_name))


def predict(img_no, plot_result):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    img = imgs_validation[[img_no], ]
    msk = msks_validation[[img_no], ]
    
    #TODO load onnx model in ngraph
    if onnx:
        onnx_protobuf = onnx.load('/data/Healthcare_app/output/unet_model_for_decathlon_100_iter.onnx')
        ng_models = import_onnx_model(onnx_protobuf)
        ng_model = ng_models[0]
        runtime = ng.runtime(backend_name='CPU')
        unet = runtime.computation(ng_model['output'], *ng_model['inputs'])
        
        start_time = time.time()
        pred_mask= unet(img)[0]
        print ("Time for prediction ngraph: ", '%.0f'%((time.time()-start_time)*1000),"ms")

    else:
        start_time = time.time()
        pred_mask = model.predict(img, verbose=0, steps=None)
        #print ("Time for prediction TF: ", '\033[1m %.0f \033[0m'%((time.time()-start_time)*1000),"ms")
       	end_time = (time.time()-start_time)*1000 
        print(end_time)
    plotDiceScore(img_no,img,msk,pred_mask,plot_result, round(end_time))
    return end_time

indicies_validation = [40, 63, 43, 55, 99, 101, 19, 46] #[40]
val_id = 1
infer_time_start = time.time()
progress_file_path = os.path.join(png_directory, "i_progress.txt")
infer_time = 0
for idx in indicies_validation:
    infer_time_idx = predict(idx, plot_result=True)
    if val_id > 2:
        infer_time += infer_time_idx
    #print((time.time()-infer_time_start)*1000/val_id)
    progressUpdate(progress_file_path, time.time()-infer_time_start, val_id, len(indicies_validation)-1) 
    val_id += 1


total_time = round(time.time() - infer_time_start, 2)
stats = {}
stats['time'] = str(total_time)
stats['frames'] = str(val_id - 2)
stats['fps'] = str( round((val_id - 2) / total_time , 2))
with open(os.path.join(png_directory, 'stats.json'), 'w') as json_file:
    json.dump(stats, json_file)
