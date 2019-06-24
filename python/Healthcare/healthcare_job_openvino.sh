
# Prevent error and output files from being saved to DevCloud
#PBS -e /dev/null
#PBS -o /dev/null

cd $PBS_O_WORKDIR
DEVICE=$1
RESULTS=$2

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
fi

if [ "$DEVICE" = "MYRIAD" ] || [ "$DEVICE" = "HDDL" ]; then
    FP_MODEL="FP16"
else
    FP_MODEL="FP32"
fi

    
# Running the object detection code
SAMPLEPATH=$PBS_O_WORKDIR
python3 healthcare_openvino.py     -d $DEVICE \
                                   -IR output/IR_models/${FP_MODEL}/saved_model \
                                   -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so \
                                   -r $RESULTS
