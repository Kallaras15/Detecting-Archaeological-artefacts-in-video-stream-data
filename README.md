# Detecting Archaeological artefacts in video stream data
### 28 March 2025  
### Leiden University – Computation Archaeology Research Group

Project developed as part of the Introduction to Machine Learning and Artificial Intelligence in Archaeology (MLA) course


# Basic Information # 
This project explores the use of YOLOv8n, a lightweight object detection model, to identify ceramic sherds in frames extracted from archaeological video recordings.
 All work, from data preparation and annotation to training and testing, was carried out locally on a standard computer.

<img src=https://github.com/user-attachments/assets/06362797-25ff-44b5-9266-ce53b85c2f5f width="350">


The directory contains the full project report, with detailed summaries of model performance, hyperparameter adjustments, and prediction analysis, as well as the Python scripts used for frame extraction and generating predictions.

# Environment Setup
To ensure that all functions run properly in your Python environment (in this case Spyder), make sure that the YOLOv8n library and the OpenCV library are properly installed.

You may install these in any terminal or command prompt associated with your Python setup (e.g., system terminal, virtual environment, IDE terminal).
YOLOv8n: Run the following command --> pip install ultralytics
OpenCV: Run the following command -->  pip install opencv-python


Finally, verify that all file paths used in the scripts correctly point to your video files, datasets, and output directories.
