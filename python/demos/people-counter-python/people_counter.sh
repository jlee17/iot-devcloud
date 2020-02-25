
#The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

#people_counter script writes output to a file inside a directory. We make sure that this directory exists.
#The output directory is the first argument of the bash script
mkdir -p $1
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4
THRESHOLD=$5

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.1
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/BSP/a10_1150_sg1/linux64/lib
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP11_YoloV3_ELU.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

SAMPLEPATH=${PBS_O_WORKDIR}
if [ "$FP_MODEL" = "FP32" ]; then
    MODELPATH=${SAMPLEPATH}/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml
else
    MODELPATH=${SAMPLEPATH}/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml
fi

#Running the people counter code
python3 people_counter.py   -m ${MODELPATH} \
                            -i ${INPUT_FILE} \
                            -o ${OUTPUT_FILE} \
                            -d ${DEVICE} \
                            -pt ${THRESHOLD}
