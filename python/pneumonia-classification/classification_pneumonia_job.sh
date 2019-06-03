
# The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR

OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
# Traffic detection script writes output to a file inside a directory. We make sure that this directory exists.
# The output directory is the first argument of the bash script
if [ "$2" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/altera/aocl-pro-rte/aclrte-linux64/
    source /opt/fpga_support_files/setup_env.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
fi

# Running the traffic detection code
SAMPLEPATH=${PBS_O_WORKDIR}
echo ${1}
python3 classification_pneumonia.py  -m models/$FP_MODEL/model.xml  \
                                           -i /validation_images/PNEUMONIA/*.jpeg \
                                           --output_dir $OUTPUT_FILE \
                                           -d $DEVICE
                                
