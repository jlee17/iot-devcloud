
# The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
NUM_VIDEOS=$4

# Traffic monitor script writes output to a file inside a directory. We make sure that this directory exists.
#  The output directory is the first argument of the bash script
mkdir -p $1

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    #source /opt/fpga_support_files/setup_env.sh
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R3_PV_PL1_FP16_MobileNet_Clamp.aocx
fi

if [ "$FP_MODEL" = "FP32" ]; then
    config_file="conf_fp32.txt"  
else
    config_file="conf_fp16.txt"
fi

SAMPLEPATH=$PBS_O_WORKDIR
# Traffic monitor takes 3 inputs, which are passed in as arguments to the bash script. 
#  -o : output directory of the videos
#  -d : device to use for inference
#  -c : conf file to use
#  -n : number of videos to process

python3 store_traffic_monitor.py  -d $DEVICE \
                                    -m ${SAMPLEPATH}/models/mobilenet-ssd/${FP_MODEL}/mobilenet-ssd.xml \
                                    -l labels.txt \
                                    -e /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so \
                                    -lp false \
                                    -o $OUTPUT_FILE \
                                    -c $config_file \
                                    -n $NUM_VIDEOS
