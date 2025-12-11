# =============================================================================
#                       AI & ML in archaeology
#                            2024 - 2025 
# =============================================================================
# Student Name: Konstantinos Kallaras (s4372603)
# Email: kallaras.dinos@gmail.com
# =============================================================================
#
#
#
# =============================================================================
#                              LIBRARIES
# =============================================================================

import cv2
import os
from ultralytics import YOLO

# =============================================================================
#                         FRAMES EXTRACTION 
# =============================================================================

def extract_frames(video_paths, output_folder):
    '''
This function is created to extract frames from multiple videos 
and save them to specified folders.

    Parameters
    ----------
    video_paths : 
    List of paths to video files.
    
    output_folder : str
        Path to the folder where frames will be saved.
    '''
    # Creates the folder if it does not exist.
    os.makedirs(output_folder, exist_ok=True) 

    # Loop through each video
    for video_path in video_paths:
        # Extract the video name (without the extension) to use as a prefix
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Open the video file
        cam = cv2.VideoCapture(video_path)

        # Initialize frame counter
        currentframe = 0

        while True:
            # Read the frame
            ret, frame = cam.read()

            # Check if the frame was successfully read
            if not ret:
                break  # Exit loop when no more frames

            # Create a unique filename for each frame using the video name and frame number
            frame_filename = os.path.join(output_folder, f"{video_name}_frame{currentframe}.jpg")
            print(f"Creating... {frame_filename}")

            # Save the frame as an image
            cv2.imwrite(frame_filename, frame)

            # Increment the frame counter
            currentframe += 1

        # Release the video file
        cam.release()

    print("All videos processed.")
    
# =============================================================================
#                      PATHS FOR EACH VIDEO
# =============================================================================

# Paths of the videos we want to extract the frams from
video_paths_a = [
    r"C:\yolov8n\MLA_Videos\Videos A\A1.mp4",
    r"C:\yolov8n\MLA_Videos\Videos A\A2.mp4",
    r"C:\yolov8n\MLA_Videos\Videos A\A3.mp4",
    "C:\yolov8n\MLA_Videos\Videos A\A4.mp4"
    ]
# Path to the folder the frames from the videos above will be saved
output_folder_a = r"C:\yolov8n\MLA_frames from videos\Frames A"

# Paths of the videos we want to extract the frams from
video_paths_b = [
    r"C:\yolov8n\MLA_Videos\Videos B\B3.mp4",
    r"C:\yolov8n\MLA_Videos\Videos B\B5.mp4",
    ]

# Path to the folder the frames from the videos above will be saved
output_folder_b = r"C:\yolov8n\MLA_frames from videos\Frames B"

# Paths of the videos we want to extract the frams from
video_paths_c = [
    r"C:\yolov8n\MLA_Videos\Videos C\C14.mp4",
    r"C:\yolov8n\MLA_Videos\Videos C\C10.mp4",
    r"C:\yolov8n\MLA_Videos\Videos C\C11.mp4"
    ]

# Path to the folder the frames from the videos above will be saved
output_folder_c = r"C:\yolov8n\MLA_frames from videos\Frames C"

# Paths of the videos we want to extract the frams from
video_paths_d = [
    r"C:\yolov8n\MLA_Videos\Videos D\D21.mp4",
    r"C:\yolov8n\MLA_Videos\Videos D\D20.mp4",
    ]

# Path to the folder the frames from the videos above will be saved
output_folder_d = r"C:\yolov8n\MLA_frames from videos\Frames D"

# =============================================================================
#              CALLING THE FUNCTION FOR EACH VIDEO TYPE
# =============================================================================


# Examples
video_a = extract_frames(video_paths_a, output_folder_a)
video_b = extract_frames(video_paths_b, output_folder_b)
video_c = extract_frames(video_paths_c, output_folder_c)
video_d = extract_frames(video_paths_d, output_folder_d)


# =============================================================================
#                   YOLOV8n TRAINING & VALIDATION
# =============================================================================


def yolov8n_train_val (learning_rate, epochs, batch_size, data_path, imgsz=640, augmentation = True):
    '''
    Train and validate function for YOLOv8 model on the given dataset.
    
    Parameters
    ----------
    learning_rate : float
        The initial learning rate for training.
    epochs : int
        The number of epochs (iterations over the entire dataset).
    batch_size : int
        The number of samples per batch for training.
    data_path : str
        The path to the YAML file containing the dataset configuration, including paths to training and validation data.
    imgsz : int, optional
       The image size to which input images will be resized during training.  The default is 640.

    Returns
    -------
    validation_results : dict
        A dictionary containing the validation results.
    train_results : dict
        A dictionary containing the training results, including metrics like loss, precision, recall, and mAP.

    '''
    # Initializing the YOLO model with a pre-trained model
    model = YOLO('yolov8n.pt')

    # Training the model
    train_results = model.train(
        data=data_path,   # Path to the dataset YAML file
        epochs=epochs,    # Number of epochs for training
        batch=batch_size,          # Batch size the model will be trained on
        lr0=learning_rate,  # Learning rate
        imgsz=imgsz,      # Image size (it will always be 640 in this case)
        augment=augmentation      # Augmentation of the pictures because we have a small dataset
    )

    # Validation after training
    validation_results = model.val(data=data_path)
    
    return validation_results, train_results

# =============================================================================
#                        HYPERPARAMETERS
# =============================================================================

epochs=20
batch_size = 8
learning_rate=0.0001 
data_path=r"C:\yolov8n\Dataset with results\dataset\somethingnew.yaml"


# Calling the function
validation_results, train_results = yolov8n_train_val(learning_rate, epochs, batch_size, data_path, imgsz=640)


# =============================================================================
#                       PREDICTIONS
# =============================================================================

def yolov8n_pred_model(model_path, test_images_path, confidence_threshold=0.5, save_results=True):
    '''
    Function to test the trained YOLOv8 model on a set of images and return the predictions.

    It loads a trained YOLOv8 model from the provided path, makes predictions on the 
    test images. It also filters predictions based on a confidence threshold.
    The predictions are saved  to disk by default in the `runs` directory under 
    the latest experiment folder.

    Parameters
    ----------
    model_path : str
        The file path to the trained YOLOv8 model weights.
    test_images_path : str
        The directory path where the images  we are going to test are stored.
    confidence_threshold : float
        The confidence threshold for predictions to be considered valid. The default is 0.5.
    save_results : bool, optional
        Whether or not to save the prediction results. Default is True. If False,
        no results will be saved to disk.
    '''

    model = YOLO(model_path)
    # Make predictions on the test images
    model.predict(source=test_images_path, conf=confidence_threshold, save=save_results)

# =============================================================================
#                        HYPERPARAMETERS
# =============================================================================

test_images_path = r"C:\yolov8n\Prediction frames\Frames C"
model_path = r"C:\yolov8n\Dataset with results\results\Best train overall\train8\weights\best.pt"
confidence_threshold=0.2
save_results=True

# Calling the function
yolov8n_pred_model (model_path, test_images_path, confidence_threshold, save_results)


