
cd $PBS_O_WORKDIR

OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.1
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/BSP/a10_1150_sg1/linux64/lib
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP11_AlexNet_GoogleNet_Generic.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

SAMPLEPATH=${PBS_O_WORKDIR}
echo ${1}
pip3 install Pillow
python3 classification_pneumonia.py -m models/$FP_MODEL/model.xml  \
                                    -i /validation_images/PNEUMONIA/*.jpeg \
                                    -o $OUTPUT_FILE \
                                    -d $DEVICE
                                
