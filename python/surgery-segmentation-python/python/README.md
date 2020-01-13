# python
This directory contains python scripts either used by the Jupyter Notebook [Robotic Surgery Demo.ipynb](../Robotic Surgery Demo.ipynb) in the parent directory or in preperation of the data used by the notebook.

* figures.py

  Generates a figure showing results of the OpenVINO model that is displayed in the Jupyter Notebook.

* img_to_video.py

  Converts a folder of images into a video file.
  The video_length variable on line 38 needs to be modified to set the number of frames in the video.

  ```bash
  python3 img_to_video.py -p <path_to_image_folder>
  ```

* models.py

  Loads the original PyTorch models.
  Line 119 was modified to use **softmax** instead of **log_softmax** since onnx export wouldn't work with the original source.

* pytorch_infer.py

  Performs inference using PyTorch.

* pytorch_to_onnx.py

  Converts the PyTorch model to an ONNX model.

* segmentation.py

  Performs binary segmentation of medical instruments.

* segmentation_parts.py

  Performs segmentation of medical instrument parts.

* utils.py

  Contains utility methods.
