### Object Detection

This advanced sample builds on the concepts introduced in the Flaw Detector Sample. The Object Detection Sample introduces the powerful capabilities of the Intel® Distribution of OpenVINO™ toolkit to perform object detection using pre-trained deep learning models on the Intel® Core™ i5-6500TE,  Intel® Xeon® Processor E3-1268L v5, Intel® HD Graphics 530, or Intel® Movidius™ NCS.

The Object Detection Sample provides a short video clip of cars. The sample presents the video frame-by-frame to the Inference Engine (IE) which subsequently uses an optimized trained neural network, mobilenet-ssd, to detect the vehicles. This public model is a mobilenet neural network (SSD – Single Shot MultiBox Detector framework) that has been pre-trained to detect objects in a video clip.


The sample illustrates two key CV architecture features:

1. Model Optimizer – A cross platform command line tool that accepts pre-trained deep learning models and optimizes them for performance using conservative topology transformations. It performs static model analysis and adjusts deep-learning models for optimal execution on target devices. NOTE: Users must run new pre-trained models through the Model Optimizer before using them. The instructions and diagrams here do not detail this step. To find out more about the Model Optimizer, visit Code Samples.

2. Inference - The process of using a trained neural network to interpret data, such as objects in images and video clips.
