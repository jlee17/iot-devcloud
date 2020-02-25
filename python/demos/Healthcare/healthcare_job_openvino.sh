
# Prevent error and output files from being saved to DevCloud
#PBS -e /dev/null

cd $PBS_O_WORKDIR
RESULTS=$1
DEVICE=$2

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.1
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/BSP/a10_1150_sg1/linux64/lib
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP11_ResNet_VGG.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

if [ "$DEVICE" = "MYRIAD" ] || [ "$DEVICE" = "HDDL" ] || [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    FP_MODEL="FP16"
else
    FP_MODEL="FP32"
fi

    
# Running the object detection code
SAMPLEPATH=$PBS_O_WORKDIR
python3 healthcare_openvino.py     -d $DEVICE \
                                   -IR output/IR_models/${FP_MODEL}/saved_model \
                                   -r $RESULTS
