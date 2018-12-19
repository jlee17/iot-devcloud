OPENVINO_FLAGS=-I/opt/intel/computer_vision_sdk/opencv/include/ \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/common/ \
	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/src/extension/ \
	    -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 \
	    -L$(INTEL_CVSDK_DIR)/opencv/lib \
            -L/data/reference-sample-data/extension/ \
	    -lgflags -linference_engine -ldl -lpthread -lcpu_extension  \
            -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lgflags \


#ORIG_OPENVINO_FLAGS=-I/opt/intel/computer_vision_sdk/opencv/include/ \
#            -I/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/include/ \
#            -I/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/include/cpp \
#            -L/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 \
#            -linference_engine -ldl -lpthread -lcpu_extension \
#            -L/opt/intel/computer_vision_sdk/opencv/lib \
#	    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lgflags \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/ \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/common/ \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/common/format_reader/ \
#	    -I$(INTEL_CVSDK_DIR)/opencv/include \
#	    -I/usr/local/include \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/thirdparty/gflags/include \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include/cpp \
#	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/extension \
#	    -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/bin/intel64/Release/lib \
#	    -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 \
#	    -L/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build/intel64/Release/lib \
#	    -L$(INTEL_CVSDK_DIR)/opencv/lib -ldl -linference_engine -lopencv_highgui -lopencv_core \
#	    -lopencv_imgproc -lopencv_videoio -lgflags_nothreads -lopencv_imgcodecs -lopencv_imgcodecs -lcpu_extension

all: face_detector 


face_detector: main.cpp detectors.cpp
	g++ -fPIE -O3 -c -o detectors.o --std=c++11 detectors.cpp -fopenmp -I. -DUSE_OPENCV ${OPENVINO_FLAGS} -I./
	g++ -fPIE -O3 -o face_detector --std=c++11 main.cpp -fopenmp -I. -DUSE_OPENCV ${OPENVINO_FLAGS} -I./ detectors.o

#ROIwriter: ROIwriter.cpp 
#	g++ -fPIE -O0 -g -o ROIwriter --std=c++11 ROIwriter.cpp -I. \
#            -I/opt/intel/computer_vision_sdk/opencv/include/ \
#            -L/opt/intel/computer_vision_sdk/opencv/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lgflags
#
#ROIviewer: ROIviewer.cpp
#	g++ -fPIE -O0 -g -o ROIviewer --std=c++11 ROIviewer.cpp -I. \
#            -I/opt/intel/computer_vision_sdk/opencv/include/ \
#            -L/opt/intel/computer_vision_sdk/opencv/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lgflags
#
