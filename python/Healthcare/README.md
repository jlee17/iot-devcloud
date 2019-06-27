### Brain Tumor Segmentation Sample

This example implements a an inference engine based on the U-Net architecture for brain tumor segmentation in MRI scans. The code demonstrates several approaches. First, a stock TensorFlow implementation is presented. Next, the same implementation is executed with Intel-optimized TensorFlow backed by MKL-DNN. Finally, an alternative implementation is laid out using the Intel® Distribution of OpenVINO™ toolkit. The latter allows you to use for inference not only Intel CPUs, but also Intel HD Graphics, Intel Neural Compute Stick 2 and HDDL-R, and Intel FPGAs (HDDL-F).

