# About the IOT Demo Catalog

The IoT DevCloud by Intel enables developers to develop and test computer vision (CV) and deep learning applications on a variety of Intel® processors and accelerator offerings. This IoT Demo Catalog contains several CV-based reference sample applications ("demos") based on the <a href="https://software.intel.com/en-us/openvino-toolkit/documentation/code-samples">Intel® Distribution of the OpenVINO™ toolkit</a>. These demos run in instances of Jupyter* Notebooks on the IoT DevCloud by Intel.

What you should know before running the demos:

* Each demo runs in a Jupyter Notebook environment on a development server based on an Intel® Xeon® Scalable processor.
* To execute inference on edge compute servers, your Jupyter Notebooks submit scripts into a job queue.
* Your home directory on the development server is network-shared between the development server and edge compute servers.

Figure 1 illustrates the process flow involved in job submission.

<img alt="Job Queue Flow" src="assets/iot-dev-cloud-workflow-diagram.svg" width="100%" />
