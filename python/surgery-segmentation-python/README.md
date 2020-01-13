# MICCAI 2017 Robotic Instrument Segmentation

![Robotic Instrument Challenge](./figures/segmentation.gif)


The code here refers to the winning solution by Alexey Shvets, Alexander Rakhlin, Alexandr A. Kalinin, and Vladimir Iglovikov in the [MICCAI 2017 Robotic Instrument Segmentation Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/). This notebook has been modified from the original found on [GitHub](https://github.com/ternaus/robot-surgery-segmentation/blob/master/Demo.ipynb) which was provided with an [MIT license](https://github.com/ternaus/robot-surgery-segmentation/blob/master/LICENSE). The data files necessary to run this notebook are included and can also be downloaded from 
[MICCAI 2017 Robotic Instrument Segmentation Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/Data/). The files **unet11_binary_20.zip**, **unet11_instruments_20.zip** and **unet11_parts_20.zip** should be downloaded from the [Google Drive](https://drive.google.com/drive/folders/13e0C4fAtJemjewYqxPtQHO6Xggk7lsKe) and extracted to the models/original directory.

![TernausNet](./figures/TernausNet.png)

## 0. Setup

Import dependencies for the notebook by running the following cells


```python
import os
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.core.display import HTML

from openvino.inference_engine import IECore, IENetwork
sys.path.insert(0, os.path.join(Path.home(), 'Reference-samples/iot-devcloud'))
from demoTools.demoutils import *
from python.utils import create_script, mask_overlay
```


```python
try: 
    import torch
except:
    print("Installing PyTorch")
    !{sys.executable} -m pip install torch
    import torch
```


```python
try: 
    from torchvision import transforms
except:
    print("Installing TorchVision")
    !{sys.executable} -m pip install torchvision
    from torchvision import transforms
```

## 1. Inference with PyTorch (Optional)
Run the following cells to perform inference with the PyTorch model on an [Intel Core i5-6500TE](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-) using the code in [pytorch_infer.py](python/pytorch_infer.py). **Wait for the progress bar to complete before running the following cells to ensure inference is complete.**


```python
# Create script to run pytorch_infer.py
create_script("generated/pytorch_infer.sh",
              "python3 python/pytorch_infer.py")

# Run the script
job_id_infer = !qsub generated/pytorch_infer.sh -l nodes=1:idc001skl:tank-870:i5-6500te -N seg_core -e logs/ -o logs/
if job_id_infer:
    print(job_id_infer[0])
    progressIndicator('results/', job_id_infer[0]+'.txt', "Inference", 0, 100)
```

    9366.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### View the original image and the inference results.


```python
image = cv2.imread("generated/input.png")
mask = cv2.imread("generated/mask.png")
plt.figure(1, figsize=(15, 15))
plt.subplot(121)
plt.axis('off')
plt.title("Input Image")
plt.imshow(image)
plt.subplot(122)
plt.axis('off')
plt.title("Segmentation")
plt.imshow(mask_overlay(image, mask));
```


![png](Robotic%20Surgery%20Demo_files/Robotic%20Surgery%20Demo_8_0.png)


## 2. Converting to ONNX
The ONNX models need to be generated from the original PyTorch models to be used with OpenVINO, do this by running [pytorch_to_onnx.py](python/pytorch_to_onnx.py). **Wait for the progress bar to complete before running the following cells to ensure all ONNX models have been generated.**


```python
# Create script to run pytorch_to_onnx.py
create_script("generated/pytorch_to_onnx.sh",
              "python3 python/pytorch_to_onnx.py")

# Run the script
job_id_onnx = !qsub generated/pytorch_to_onnx.sh -l nodes=1:idc001skl:tank-870:i5-6500te -N seg_core -e logs/ -o logs/
if job_id_onnx:
    print(job_id_onnx[0])
    progressIndicator('results/', job_id_onnx[0]+'.txt', "Inference", 0, 100)
```

    9367.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


## 3. Inference with OpenVINO

In this section, we will walk through an example to showcase the Python API of the OpenVINO inference engine. The first step will be to run the model optimizer on the pre-trained model to produce an intermediate representation (IR) which will be used by OpenVINO's inference engine. 

For this example, the model is trained using PyTorch and saved in the ONNX format. More details on converting to the onnx format can be found [here](https://pytorch.org/docs/stable/onnx.html). Descriptions of each parameter for the model optimizer script can be found in the OpenVINO documentation [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html).

### Running the OpenVINO Model Optimizer


```python
!python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model "models/onnx/surgical_tools.onnx" \
    --output_dir models/ov/FP32/ \
    --data_type FP32 \
    --move_to_preprocess \
    --scale_values "[0.229, 0.224, 0.225]" \
    --mean_values "[0.485, 0.456, 0.406]"
```

    Model Optimizer arguments:
    Common parameters:
    	- Path to the Input Model: 	/home/u33133/surgery-segmentation-python-test/models/onnx/surgical_tools.onnx
    	- Path for generated IR: 	/home/u33133/surgery-segmentation-python-test/models/ov/FP32/
    	- IR output name: 	surgical_tools
    	- Log level: 	ERROR
    	- Batch: 	Not specified, inherited from the model
    	- Input layers: 	Not specified, inherited from the model
    	- Output layers: 	Not specified, inherited from the model
    	- Input shapes: 	Not specified, inherited from the model
    	- Mean values: 	[0.485, 0.456, 0.406]
    	- Scale values: 	[0.229, 0.224, 0.225]
    	- Scale factor: 	Not specified
    	- Precision of IR: 	FP32
    	- Enable fusing: 	True
    	- Enable grouped convolutions fusing: 	True
    	- Move mean values to preprocess section: 	True
    	- Reverse input channels: 	False
    ONNX specific parameters:
    Model Optimizer version: 	2019.3.0-375-g332562022
    
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/u33133/surgery-segmentation-python-test/models/ov/FP32/surgical_tools.xml
    [ SUCCESS ] BIN file: /home/u33133/surgery-segmentation-python-test/models/ov/FP32/surgical_tools.bin
    [ SUCCESS ] Total execution time: 3.88 seconds. 



```python
!python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model "models/onnx/surgical_tools_parts.onnx" \
    --output_dir models/ov/FP32/ \
    --data_type FP32 \
    --move_to_preprocess \
    --scale_values "[0.229, 0.224, 0.225]" \
    --mean_values "[0.485, 0.456, 0.406]"
```

    Model Optimizer arguments:
    Common parameters:
    	- Path to the Input Model: 	/home/u33133/surgery-segmentation-python-test/models/onnx/surgical_tools_parts.onnx
    	- Path for generated IR: 	/home/u33133/surgery-segmentation-python-test/models/ov/FP32/
    	- IR output name: 	surgical_tools_parts
    	- Log level: 	ERROR
    	- Batch: 	Not specified, inherited from the model
    	- Input layers: 	Not specified, inherited from the model
    	- Output layers: 	Not specified, inherited from the model
    	- Input shapes: 	Not specified, inherited from the model
    	- Mean values: 	[0.485, 0.456, 0.406]
    	- Scale values: 	[0.229, 0.224, 0.225]
    	- Scale factor: 	Not specified
    	- Precision of IR: 	FP32
    	- Enable fusing: 	True
    	- Enable grouped convolutions fusing: 	True
    	- Move mean values to preprocess section: 	True
    	- Reverse input channels: 	False
    ONNX specific parameters:
    Model Optimizer version: 	2019.3.0-375-g332562022
    
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/u33133/surgery-segmentation-python-test/models/ov/FP32/surgical_tools_parts.xml
    [ SUCCESS ] BIN file: /home/u33133/surgery-segmentation-python-test/models/ov/FP32/surgical_tools_parts.bin
    [ SUCCESS ] Total execution time: 4.21 seconds. 



```python
!python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model "models/onnx/surgical_tools.onnx" \
    --output_dir models/ov/FP16/ \
    --data_type FP16 \
    --move_to_preprocess \
    --scale_values "[0.229, 0.224, 0.225]" \
    --mean_values "[0.485, 0.456, 0.406]"
```

    Model Optimizer arguments:
    Common parameters:
    	- Path to the Input Model: 	/home/u33133/surgery-segmentation-python-test/models/onnx/surgical_tools.onnx
    	- Path for generated IR: 	/home/u33133/surgery-segmentation-python-test/models/ov/FP16/
    	- IR output name: 	surgical_tools
    	- Log level: 	ERROR
    	- Batch: 	Not specified, inherited from the model
    	- Input layers: 	Not specified, inherited from the model
    	- Output layers: 	Not specified, inherited from the model
    	- Input shapes: 	Not specified, inherited from the model
    	- Mean values: 	[0.485, 0.456, 0.406]
    	- Scale values: 	[0.229, 0.224, 0.225]
    	- Scale factor: 	Not specified
    	- Precision of IR: 	FP16
    	- Enable fusing: 	True
    	- Enable grouped convolutions fusing: 	True
    	- Move mean values to preprocess section: 	True
    	- Reverse input channels: 	False
    ONNX specific parameters:
    Model Optimizer version: 	2019.3.0-375-g332562022
    
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/u33133/surgery-segmentation-python-test/models/ov/FP16/surgical_tools.xml
    [ SUCCESS ] BIN file: /home/u33133/surgery-segmentation-python-test/models/ov/FP16/surgical_tools.bin
    [ SUCCESS ] Total execution time: 3.93 seconds. 



```python
!python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model "models/onnx/surgical_tools_parts.onnx" \
    --output_dir models/ov/FP16/ \
    --data_type FP16 \
    --move_to_preprocess \
    --scale_values "[0.229, 0.224, 0.225]" \
    --mean_values "[0.485, 0.456, 0.406]"
```

    Model Optimizer arguments:
    Common parameters:
    	- Path to the Input Model: 	/home/u33133/surgery-segmentation-python-test/models/onnx/surgical_tools_parts.onnx
    	- Path for generated IR: 	/home/u33133/surgery-segmentation-python-test/models/ov/FP16/
    	- IR output name: 	surgical_tools_parts
    	- Log level: 	ERROR
    	- Batch: 	Not specified, inherited from the model
    	- Input layers: 	Not specified, inherited from the model
    	- Output layers: 	Not specified, inherited from the model
    	- Input shapes: 	Not specified, inherited from the model
    	- Mean values: 	[0.485, 0.456, 0.406]
    	- Scale values: 	[0.229, 0.224, 0.225]
    	- Scale factor: 	Not specified
    	- Precision of IR: 	FP16
    	- Enable fusing: 	True
    	- Enable grouped convolutions fusing: 	True
    	- Move mean values to preprocess section: 	True
    	- Reverse input channels: 	False
    ONNX specific parameters:
    Model Optimizer version: 	2019.3.0-375-g332562022
    
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/u33133/surgery-segmentation-python-test/models/ov/FP16/surgical_tools_parts.xml
    [ SUCCESS ] BIN file: /home/u33133/surgery-segmentation-python-test/models/ov/FP16/surgical_tools_parts.bin
    [ SUCCESS ] Total execution time: 4.00 seconds. 


## 4. Inference on the edge

All the code up to this point has been run within the Jupyter Notebook instance running on a development node based on an Intel Xeon Scalable processor, where the Notebook is allocated a single core. We will run the workload on other edge compute nodes represented in the IoT DevCloud. We will send work to the edge compute nodes by submitting the corresponding non-interactive jobs into a queue. For each job, we will specify the type of the edge compute server that must be allocated for the job.

The job file is written in Bash, and will be executed directly on the edge compute node. For this example, we have written the job file for you in the notebook. It performs the classification using the script "segmentation.sh".


```python
%%writefile generated/segmentation.sh

# The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR


OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    #export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R3_PV_PL1_FP16_MobileNet_Clamp.aocx
    #aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
    #aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP16_ResNet_SqueezeNet_VGG_ELU.aocx

fi


python3 python/segmentation_parts.py  -m ${FP_MODEL} \
                               -i data/${INPUT_FILE} \
                               -d ${DEVICE} \
                               -o ${OUTPUT_FILE}
```

    Overwriting generated/segmentation.sh


### 4.1 Understand how jobs are submitted into the queue

Now that we have the job script, we can submit the jobs to edge compute nodes. In the IoT DevCloud, you can do this using the `qsub` command.
We can submit the job to 5 different types of edge compute nodes simultaneously or just one node at at time.

There are five options of `qsub` command that we use for this:
- `-l` : this option lets us select the number and the type of nodes using `nodes={node_count}:{property}`. 
- `-F` : this option lets us send arguments to the bash script. 
- `-N` : this option lets us name the job so that it is easier to distinguish between them.
- `-o` : this option lets us determine the path to be used for the standard output stream.
- `-e` : this option lets us determine the path to be used for the standard error stream.


The `-F` flag is used to pass in arguments to the job script.
The [segmentation.sh](segmentation.sh) script takes in 4 arguments:
1. the path to the directory for the output video and performance stats
2. targeted device (e.g. CPU, GPU, MYRIAD, HDDL or HETERO:FPGA,CPU)
3. the floating precision to use for inference
4. the path to the input video

The job scheduler will use the contents of `-F` flag as the argument to the job script.

If you are curious to see the available types of nodes on the IoT DevCloud, run the following optional cell.


```python
!pbsnodes | grep compnode | sort | uniq -c
```

         35      properties = idc001skl,compnode,iei,tank-870,intel-core,i5-6500te,skylake,intel-hd-530,ram8gb,net1gbe
         15      properties = idc002mx8,compnode,iei,tank-870,intel-core,i5-6500te,skylake,intel-hd-530,ram8gb,net1gbe,hddl-r,iei-mustang-v100-mx8
         17      properties = idc003a10,compnode,iei,tank-870,intel-core,i5-6500te,skylake,intel-hd-530,ram8gb,net1gbe,hddl-f,iei-mustang-f100-a10
         23      properties = idc004nc2,compnode,iei,tank-870,intel-core,i5-6500te,skylake,intel-hd-530,ram8gb,net1gbe,ncs,intel-ncs2
          6      properties = idc006kbl,compnode,iei,tank-870,intel-core,i5-7500t,kaby-lake,intel-hd-630,ram8gb,net1gbe
         13      properties = idc007xv5,compnode,iei,tank-870,intel-xeon,e3-1268l-v5,skylake,intel-hd-p530,ram32gb,net1gbe
         15      properties = idc008u2g,compnode,up-squared,grove,intel-atom,e3950,apollo-lake,intel-hd-505,ram4gb,net1gbe,ncs,intel-ncs2
          1      properties = idc009jkl,compnode,jwip,intel-core,i5-7500,kaby-lake,intel-hd-630,ram8gb,net1gbe
          1      properties = idc010jal,compnode,jwip,intel-atom,e3950,apollo-lake,intel-hd-505,ram4gb,net1gbe
          1      properties = idc011ark2250s,compnode,advantech,intel-core,i5-6442eq,skylake,intel-hd-503,ram8gb,net1gbe
          1      properties = idc012ark1220l,compnode,advantech,intel-atom,e3940,apollo-lake,intel-hd-500,ram4gb,net1gbe
          1      properties = idc013ds580,compnode,advantech,intel-atom,e3950,apollo-lake,intel-hd-505,ram2gb,net1gbe
          1      properties = idc101agg,compnode,iei,tank-870,intel-core,i5-7500t,kaby-lake,intel-hd-630,ram8gb,net1gbe
          1      properties = idc101col,compnode,iei,tank-870,intel-core,i5-7500t,kaby-lake,intel-hd-630,ram8gb,net1gbe
          1      properties = idc101rdk,compnode,iei,tank-870,intel-core,i5-6500te,skylake,intel-hd-530,ram8gb,net1gbe,ncs,intel-ncs2
          2      properties = idc101ros,compnode,iei,tank-870,intel-core,i5-7500t,kaby-lake,intel-hd-630,ram8gb,net1gbe
          3      properties = idc101xv5,compnode,iei,tank-870,intel-xeon,e3-1268l-v5,skylake,intel-hd-p530,ram32gb,net1gbe


Here, the properties describe the node, and number on the left is the number of available nodes of that architecture.

### 4.2 Job queue submission

The output of the cell is the `JobID` of your job, which you can use to track progress of a job.

**Note** You can submit all the jobs at once or follow one at a time. 

After submission, they will go into a queue and run as soon as the requested compute resources become available. 
(tip: **shift+enter** will run the cell and automatically move you to the next cell. So you can hit **shift+enter** multiple times to quickly run multiple cells).


#### Submitting to an edge compute node with an Intel Core CPU
In the cell below, we submit a job to an <a 
    href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a 
    href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel 
    Core i5-6500TE</a>. The inference workload will run on the CPU.


```python
job_id_core = !qsub generated/segmentation.sh -l nodes=1:idc001skl:tank-870:i5-6500te -F "results/ CPU FP32 short_source.mp4" -N seg_core -e logs/ -o logs/
print(job_id_core[0]) 
#Progress indicators
if job_id_core:
    progressIndicator('results/', job_id_core[0]+'.txt', "Inference", 0, 100)
```

    9368.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### Submitting to an edge compute node with Intel Xeon CPU
In the cell below, we submit a job to an <a 
    href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a 
    href="https://ark.intel.com/products/88178/Intel-Xeon-Processor-E3-1268L-v5-8M-Cache-2-40-GHz-">Intel 
    Xeon Processor E3-1268L v5</a>. The inference workload will run on the CPU.


```python
#Submit job to the queue
job_id_xeon = !qsub generated/segmentation.sh -l nodes=1:tank-870:e3-1268l-v5 -F "results/ CPU FP32 short_source.mp4" -N seg_xeon -e logs/ -o logs/
print(job_id_xeon[0])

#Progress indicator
if job_id_xeon:
    progressIndicator('results/', job_id_xeon[0]+'.txt', "Inference", 0, 100)
```

    9369.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### Submitting to an edge compute node with Intel Core CPU and using the onboard Intel GPU
In the cell below, we submit a job to an <a 
    href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core i5-6500TE</a>. The inference workload will run on the Intel® HD Graphics 530 card integrated with the CPU.


```python
#Submit job to the queue
job_id_gpu = !qsub generated/segmentation.sh -l nodes=1:tank-870:i5-6500te:intel-hd-530 -F "results/ GPU FP16 short_source.mp4" -N seg_gpu -e logs/ -o logs/
print(job_id_gpu[0])

#Progress indicator
if job_id_gpu:
    progressIndicator('results/', job_id_gpu[0]+'.txt', "Inference", 0, 100)
```

    9370.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### Submitting to an edge compute node with  IEI Mustang-F100-A10 (Intel® Arria® 10 FPGA)
In the cell below, we submit a job to an <a 
    href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core i5-6500te CPU</a> . The inference workload will run on the <a href="https://www.ieiworld.com/mustang-f100/en/"> IEI Mustang-F100-A10 </a> card installed in this node.


```python
#Submit job to the queue
job_id_fpga = !qsub generated/segmentation.sh -l nodes=1:idc003a10:iei-mustang-f100-a10 -F "results/ HETERO:FPGA,CPU FP16 short_source.mp4" -N seg_fpga -e logs/ -o logs/
print(job_id_fpga[0]) 

#Progress indicator
if job_id_fpga:
    progressIndicator('results/', job_id_fpga[0]+'.txt', "Inference", 0, 100)
```

    9371.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### Submitting to an edge compute node with Intel NCS 2 (Neural Compute Stick 2)
In the cell below, we submit a job to an <a 
    href="https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core">IEI 
    Tank 870-Q170</a> edge node with an <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel Core i5-6500te CPU</a>. The inference workload will run on an <a 
    href="https://software.intel.com/en-us/neural-compute-stick">Intel Neural Compute Stick 2</a> installed in this  node.


```python
#Submit job to the queue
job_id_myriadx = !qsub generated/segmentation.sh -l nodes=1:idc004nc2:intel-ncs2 -F "results/ MYRIAD FP16 short_source.mp4" -N seg_myriadx -e logs/ -o logs/
print(job_id_myriadx[0])

#Progress indicator
if job_id_myriadx:
    progressIndicator('results/', job_id_myriadx[0]+'.txt', "Inference", 0, 100)
```

    9372.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


#### Generating example visualzations using an Intel Core CPU
In the cell below, we submit a job to run the script [figures.py](python/figures.py) to an [IEI Tank 870-Q170](https://software.intel.com/en-us/iot/hardware/iei-tank-dev-kit-core) edge node with an [Intel Core i5-6500TE](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-). The inference workload will run on the CPU.


```python
# Create script to run figures.py
create_script("generated/figures.sh",
              "python3 python/figures.py")

# Run the script
job_id_figures = !qsub generated/figures.sh -l nodes=1:idc001skl:tank-870:i5-6500te -N figures -e logs/ -o logs/
if job_id_figures:
    print(job_id_figures[0])
    progressIndicator('results/', job_id_figures[0]+'.txt', "Inference", 0, 100)
```

    9373.v-qsvr-1.devcloud-edge



    HBox(children=(FloatProgress(value=0.0, bar_style='info', description='Inference', style=ProgressStyle(descrip…


### 4.3 Check if the jobs are done

To check on the jobs that were submitted, use the `qstat` command.

We have created a custom Jupyter widget  to get live qstat update.
Run the following cell to bring it up. 


```python
liveQstat()
```


    Output(layout=Layout(border='1px solid gray', height='300px', width='100%'))



    Button(description='Stop', style=ButtonStyle())


You should see the jobs you have submitted (referenced by `Job ID` that gets displayed right after you submit the job in step 2.3).
There should also be an extra job in the queue "jupyterhub": this job runs your current Jupyter Notebook session.

The 'S' column shows the current status. 
- If it is in Q state, it is in the queue waiting for available resources. 
- If it is in R state, it is running. 
- If the job is no longer listed, it means it is completed.

**Note**: Time spent in the queue depends on the number of users accessing the edge nodes. Once these jobs begin to run, they should take from 1 to 5 minutes to complete. 

***Wait!***

Please wait for the inference jobs complete before proceeding to the next step.

### 4.4 View Results

Once the jobs are completed, the stdout and stderr streams of each job are saved into the `logs/` folder.

We also saved the probability output and inference time for each input image in the folder `results/` for each architecture. 
We observe the results below.

#### Result on the Intel Core CPU 


```python
videoHTML('IEI Tank (Intel Core CPU)', 
          ['results/output_'+job_id_core[0]+'.mp4'], 
          'results/stats_'+job_id_core[0]+'.txt')
```




<h2>IEI Tank (Intel Core CPU)</h2>
    <p>20
 frames processed in 72.3
 seconds</p>
    <video alt="" controls autoplay muted height="480"><source src="results/output_9368.v-qsvr-1.devcloud-edge.mp4" type="video/mp4" /></video>




#### Result on the Intel Xeon CPU


```python
videoHTML('IEI Tank Xeon (Intel Xeon CPU)',
          ['results/output_'+job_id_xeon[0]+'.mp4'],
          'results/stats_'+job_id_xeon[0]+'.txt')
```




<h2>IEI Tank Xeon (Intel Xeon CPU)</h2>
    <p>20
 frames processed in 72.2
 seconds</p>
    <video alt="" controls autoplay muted height="480"><source src="results/output_9369.v-qsvr-1.devcloud-edge.mp4" type="video/mp4" /></video>




#### Result on the Intel Integrated GPU


```python
videoHTML('IEI Intel GPU (Intel Core + Onboard GPU)', 
          ['results/output_'+job_id_gpu[0]+'.mp4'],
          'results/stats_'+job_id_gpu[0]+'.txt')
```




<h2>IEI Intel GPU (Intel Core + Onboard GPU)</h2>
    <p>20
 frames processed in 182.2
 seconds</p>
    <video alt="" controls autoplay muted height="480"><source src="results/output_9370.v-qsvr-1.devcloud-edge.mp4" type="video/mp4" /></video>




#### Result on the IEI Mustang-F100-A10 (Intel® Arria® 10 FPGA)


```python
videoHTML('IEI Mustang-F100-A10',
          ['results/output_'+job_id_fpga[0]+'.mp4'],
          'results/stats_'+job_id_fpga[0]+'.txt')
```




<h2>IEI Mustang-F100-A10</h2>
    <p>20
 frames processed in 28.6
 seconds</p>
    <video alt="" controls autoplay muted height="480"><source src="results/output_9371.v-qsvr-1.devcloud-edge.mp4" type="video/mp4" /></video>




#### Visualizaiton results on the Intel Core CPU


```python
HTML('''{img}'''.format(img="<img src='generated/predictions.png'>"))
```




<img src='generated/predictions.png'>



### 4.5 Assess Performance

The total average time of each inference task is recorded in `results/{ARCH}/statsjob_id.txt`, where the subdirectory name corresponds to the architecture of the target edge compute node. Run the cell below to plot the results of all jobs side-by-side. Lower values mean better performance. Keep in mind that some architectures are optimized for the highest performance, others for low power or other metrics.


```python
arch_list = [('core', 'Intel Core\ni5-6500TE\nCPU'),
             ('xeon', 'Intel Xeon\nE3-1268L v5\nCPU'),
             ('gpu', ' Intel Core\ni5-6500TE\nGPU'),
             ('fpga', 'Intel\nFPGA')]

stats_list = []


#stats_list.append(('results/untitled.txt', 'PyTorch\nIntel Core\ni5-6500TE\nCPU'))

for arch, a_name in arch_list:
    if 'job_id_'+arch in vars():
        stats_list.append(('results/stats_'+vars()['job_id_'+arch][0]+'.txt', a_name))
    else:
        stats_list.append(('placeholder'+arch, a_name))
summaryPlot(stats_list, 'Architecture', 'Time(s)', 'Inference Engine Processing Time', 'time' )
```


![png](Robotic%20Surgery%20Demo_files/Robotic%20Surgery%20Demo_48_0.png)


## 5. Citation
    @inproceedings{shvets2018automatic,
    title={Automatic Instrument Segmentation in Robot-Assisted Surgery using Deep Learning},
    author={Shvets, Alexey A and Rakhlin, Alexander and Kalinin, Alexandr A and Iglovikov, Vladimir I},
    booktitle={2018 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
    pages={624--628},
    year={2018}
    }


```python

```
