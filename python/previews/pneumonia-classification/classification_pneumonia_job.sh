
cd $PBS_O_WORKDIR

OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3

if [ "$2" = "HETERO:FPGA,CPU" ]; then
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2019R3_PV_PL2_FP16_AlexNet_GoogleNet_InceptionV1_SSD300_Generic.aocx
fi

SAMPLEPATH=${PBS_O_WORKDIR}
echo ${1}
pip3 install Pillow
python3 classification_pneumonia.py -m models/$FP_MODEL/model.xml  \
                                    -i /validation_images/PNEUMONIA/*.jpeg \
                                    -o $OUTPUT_FILE \
                                    -d $DEVICE
                                
