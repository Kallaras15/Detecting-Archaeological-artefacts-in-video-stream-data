# Detecting Archaeological artefacts in video stream data#
### 28 March 2025  
### Leiden University – Computation Archaeology Research Group

Project developed as part of the Introduction to Machine Learning and Artificial Intelligence in Archaeology (MLA) course


# Basic Information # 
This project explores the use of YOLOv8n, a lightweight object detection model, to identify ceramic sherds in frames extracted from archaeological video recordings.
All work—data preparation, annotation, training, and testing—was carried out locally on a standard computer.

<img src=https://github.com/user-attachments/assets/06362797-25ff-44b5-9266-ce53b85c2f5f width="350">


The directory contains 2 files:
- Assignment 3: contains all model implementations, the Code developed for this project, and the yolov8n.pt file, which is the base pre-trained model, before training with our dataset. A report (PDF) is also included with a detailed summary of model performance in each run, changes in hyperparameters, and prediction analysis. 

- Predictions - Assignment 3: Contains predictions made on four types of video frames (A to D), each representing increasing levels of difficulty. The predictions were made using the weights from the best-performing run of each model. There are 20 indicative predictions per video type for each model. The models were tested on a broader set of images. The prediction frames were not used during training, validation, or prior testing, ensuring an unbiased evaluation of model performance.
