
#The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

#restricted_zone_job script writes output to a file inside a directory. We make sure that this directory exists.
#The output directory is the first argument of the bash script
mkdir -p $1
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.1
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/BSP/a10_1150_sg1/linux64/lib
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP11_MobileNet_Clamp.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

SAMPLEPATH=${PBS_O_WORKDIR}
if [ "$FP_MODEL" = "FP32" ]; then
  MODELPATH=${SAMPLEPATH}/models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml
else
  MODELPATH=${SAMPLEPATH}/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml
fi

python3 restricted_zone_notifier.py -m ${MODELPATH}\
                                    -i ${INPUT_FILE}\
                                    -o ${OUTPUT_FILE}\
                                    -d ${DEVICE}

g++ -std=c++14 ROI_writer.cpp -o ROI_writer_${PBS_JOBID}  -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui  -fopenmp -I/opt/intel/openvino/opencv/include/ -L/opt/intel/openvino/opencv/lib/

#Rendering the output video
SKIPFRAME=1
RESOLUTION=0.5
./ROI_writer_${PBS_JOBID} $INPUT_FILE $OUTPUT_FILE $SKIPFRAME $RESOLUTION
rm -f ROI_writer_${PBS_JOBID}
