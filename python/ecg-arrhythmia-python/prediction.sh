
cd $PBS_O_WORKDIR

DEVICE=$1

if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
  # Environment variables and compilation for edge compute nodes with FPGAs
  source /opt/intel/init_openvino.sh
  aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R3_PV_PL1_FP16_MobileNet_Clamp.aocx
fi

python3 inference.py -d ${DEVICE}
