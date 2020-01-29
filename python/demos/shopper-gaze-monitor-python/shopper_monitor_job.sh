
#The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

#shopper_monitor_job script writes output to a file inside a directory. We make sure that this directory exists.
#The output directory is the first argument of the bash script
mkdir -p $1
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4

if [ $DEVICE = "HETERO:FPGA,CPU" ]; then
    #Environment variables and compilation for edge compute nodes with FPGAs
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R3_PV_PL1_FP16_MobileNet_Clamp.aocx
fi

SAMPLEPATH=${PBS_O_WORKDIR}
if [ "$FP_MODEL" = "FP32" ]; then
  MODELPATH=${SAMPLEPATH}/models/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
else
  MODELPATH=${SAMPLEPATH}/models/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml
fi

if [ "$FP_MODEL" = "FP32" ]; then
  POSE_MODEL=${SAMPLEPATH}/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
else
  POSE_MODEL=${SAMPLEPATH}/models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
fi

#Running the shopper gaze monitor code
python3 shopper_gaze_monitor.py -m ${MODELPATH} \
                                -i ${INPUT_FILE} \
                                -o ${OUTPUT_FILE} \
                                -d ${DEVICE} \
                                -pm ${POSE_MODEL} \
                                -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so
