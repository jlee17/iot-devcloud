
# The default path for the job is your home directory, so we change directory to where the files are.
cd $PBS_O_WORKDIR
mkdir -p $1
# Running the object detection code
# -l /opt/intel/openvino/deployment_tools/inference_engine/samples/build/intel64/Release/lib/libcpu_extension.so \
SAMPLEPATH=$PBS_O_WORKDIR
python3 classification_sample.py  -m model/$3/crnn.xml  \
                                           -i board4.jpg \
                                           -o $1 \
                                           -d $2
                                           -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so                                           
