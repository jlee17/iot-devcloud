OPENVINO_FLAGS=-I/opt/intel/computer_vision_sdk/opencv/include/ \
            -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/include \
	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/samples/common/ \
	    -I$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/src/extension/ \
	    -L$(INTEL_CVSDK_DIR)/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 \
	    -L$(INTEL_CVSDK_DIR)/opencv/lib \
            -L/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build/intel64/Release/lib/ \
	    -lgflags -linference_engine -ldl -lpthread -lcpu_extension  \
            -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_video -lgflags \

all: face_detector 


face_detector: main.cpp detectors.cpp
	g++ -fPIE -O3 -c -o detectors.o --std=c++11 detectors.cpp -fopenmp -I. -DUSE_OPENCV ${OPENVINO_FLAGS} -I./
	g++ -fPIE -O3 -o face_detector --std=c++11 main.cpp -fopenmp -I. -DUSE_OPENCV ${OPENVINO_FLAGS} -I./ detectors.o

