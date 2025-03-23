# CricPose360: Real-Time Pose-Enhanced Cricket Shot Classification

## Overview

CricPose360 is a dataset and model pipeline designed for real-time cricket shot classification. The project introduces a novel pose-enhanced approach to improve classification accuracy by utilizing skeleton keypoints extracted from videos using the YOLOv8 model. The dataset, CricPose360, contains 3,030 annotated cricket videos featuring 10 distinct cricket shots.

## Project Structure

```bash
/CricPose360_Project
│
├── EDA.ipynb                     # Exploratory Data Analysis of the dataset
├── LICENSE                       # Project license information
├── README.md                     # Project overview and instructions
├── Resnet3Dl_model_training.ipynb # Training script for ResNet3D model
├── Video-processing.ipynb        # Video preprocessing script
├── YOLOV8-cls_fine_tuning.ipynb  # Fine-tuning YOLOv8 for classification
├── YOLOV8_Inference(fine_tuned).ipynb # Inference using fine-tuned YOLOv8 model
```

## Installation

Ensure you have Python 3.8+ installed, along with the required dependencies. You can install them via pip:

```bash
pip install -r requirements.txt
``` 

## Dataset

The CricPose360 dataset includes 3,030 annotated cricket videos divided into 10 distinct batting shots:

``` bash
1. Cover Drive
2. Defensive Shot
3. Flick
4. Hook
5. Late Cut
6. Lofted Shot
7. Pull Shot
8. Square Cut
9. Straight Drive
10. Sweep Shot
```

The dataset is split into training (70%), validation (20%), and test (10%) sets.

## Pose-Enhanced Data

Pose-enhanced data is generated using the YOLOv8 pose model to extract skeleton keypoints from the frames. The pose-enhanced videos are used to train a YOLOv8 model for shot classification.

## Methodology

Data Collection: 3,030 cricket videos, each containing a sequence of 3-7 second clips of various batting shots.

Preprocessing: The videos are preprocessed to extract frames, and pose annotations are generated using YOLOv8 pose.

Model Training: YOLOv8 is used for fine-tuning and classification. The model is trained on the pose-enhanced dataset.

Inference: Fine-tuned models are used for classifying unseen cricket shots.

## Results

Accuracy: 92.21% on the test dataset.

F1 Score: 92.29%.

Inference Speed: 16.1 ms/frame.

## Future Work

Further optimization of pose-enhanced data for improved accuracy.

Exploration of transformer-based models for classification enhancement.

Integration of ball tracking techniques to boost shot classification.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

YOLOv8: For object detection and pose estimation.

FFmpeg: For video frame extraction and reconstruction.

