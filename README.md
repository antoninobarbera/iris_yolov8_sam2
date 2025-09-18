# Multi-Algorithm Iris Recognition System

This project presents a multi-algorithm iris recognition system that leverages YOLOv8 for eye detection and SAM2 (Segment Anything 2) for precise iris and pupil segmentation. The system is designed to support a variety of feature extractors and diverse datasets for comprehensive and robust analysis.

## ABSTRACT ##
In this work, a hybrid identification system is presented that combines segmentation techniques based on YOLOv8 and Segment Anything Model 2 (SAM2) models, with machine learning approaches to enhance the accuracy and robustness of the identification process. The system employs YOLOv8 to quickly localize the iris region and SAM2 to obtain precise boundary segmentation, ensuring an accurate representation of biometric features.
The proposed hierarchical framework utilizes three classifiers: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and a feedforward neural network (FFNN), which operate on features extracted from the iris code. These classifiers efficiently narrow down the candidate set, which is subsequently verified by a Scale-Invariant Feature Transform (SIFT) algorithm that performs precise feature matching. Furthermore, the system combines the robustness of segmentation provided by YOLOv8 and SAM2 with the efficiency of lightweight classifiers and candidate set reduction, ensuring fast processing times and high performance. This solution gave accuracies of 100.0%, 99.09%, 98.33% e 99.48% for CASIA-Iris-V1, CASIA-V3-Interval, CASIA-V3-Lamp and UBIRIS V1 databases, respectively.

## Project Description

The system is capable of:
*   **Eye detection** using YOLOv8.
*   **Iris and pupil segmentation** with SAM2 for high accuracy.
*   Supporting various **feature extractors**, including classical algorithms and advanced neural networks: Daugman, ResNet101, DenseNet201, ViT (Vision Transformer), Swin Transformer, and a Hybrid approach.
*   Working with several **iris datasets**: CASIA v1, CASIA v3 Interval, CASIA v3 Lamp, UBIRIS V1, UBIRIS V2 (segmentation only), and CASIA v3 Twins (segmentation only).
    * Note: All datasets should be prepared and stored in **pickle** format for compatibility with the system. *

This project was developed as part of the Bachelor's Degree in Artificial Intelligence and Cybersecurity Engineering at the University of Enna "Kore".

**Student:** Barbera Antonino
**Academic Year:** 2024/2025

## Installation

To get started, clone the repository and install the necessary dependencies.

1.  **Clone the SAM2 repository (prerequisite for this project):**
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd SAM2
    ```
    *Note: Ensure you clone the SAM2 repository as the base for the environment.*

2. **Download SAM's checkpoints**  
   Download the checkpoints from [SAM model checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place them in the `checkpoints/` folder.


3.  **Install required libraries:**
    *   On Linux/Windows systems (make sure you are in the root folder of your project that contains `requirements.txt`):
        ```bash
        pip install -r requirements.txt
        ```
    *   On GitBash/Mac:
        ```bash
        bash prepare.sh
        ```
    *   Install SAM2 in editable mode (so that your project can import `sam2`):
        ```bash
        cd SAM2
        pip install -e .
        ```

4.  **Troubleshooting OpenCV issues (if necessary):**
    If you encounter problems with OpenCV, try uninstalling the existing version and reinstalling via `requirements.txt`:
    ```bash
    pip uninstall opencv-python
    pip install -r requirements.txt
    ```

## How to Run the Project

The system supports several operating modes:

### 1. Identification Mode

Executes the complete recognition pipeline, which includes segmentation (YOLOv8 + SAM2), normalization, feature extraction, iris code/feature vector generation, matching, and performance evaluation.

**Command:**
```bash
python hybrid_iris_recognition/main.py
```
**Output:**

* Folder `hybrid_iris_recognition/iris_images` containing original, segmented, normalized images, iris codes, keypoints, and masks.
* Folder `hybrid_iris_recognition/graph` with ROC and FAR/FRR plots.


### 2. Segmentation Mode

Performs only iris and pupil segmentation.

**Command:**
```bash
python hybrid_iris_recognition/main.py
```
*Note: Ensure that system_mode: [segmentation] is set in the `config.yaml` file.*

**Output:**
* Folder `iris_images` cwith segmented images and a report on how many images were correctly segmented.

### 3. ML Classifiers-based Algorithm

Uses extracted features to train Machine Learning classifiers such as SVM, KNN, and Neural Networks (NN).

```bash
python hybrid_iris_recognition/models_generation.py
```
**Output:**

* Folder `graph/`: Plots related to the accuracy of the trained classifiers.
* Classifier performance metrics.


### 4. SIFT-based Algorithm

Implements a recognition approach based on SIFT keypoints and matching. This method is computationally more intensive than others.

```bash
python hybrid_iris_recognition/sift_test.py
```

## Configuration  (config.yaml)

The `config.yaml`file allows customization of system parameters. Here is an example of the main parameters:

```bash
dataset_to_use: [casiav1] # Options: casiav1, ubirisv1, casia_v3_interval, casia_v3_lamp
system_mode: [identification] # Options: identification or segmentation
feature_extractor: [daugman] # Options: daugman, resnet101, densenet201, vit, swin, hybrid
sam2_type: "small" # Options for SAM2: tiny, small, large, plus
test_thresholds: [8, 16, 24, 36, 40]
graph:
  save_images: True
  save_yolo_annotation: False
```

## Output
The system generates the following types of output:

Generated Images:
* Original
* Segmented
* Normalized
* Iris code
* Keypoints
* SAM2 masks (iris/pupil)

Grafici:
* Accuracy Curve (for ML classifiers)
* FAR/FRR Curve (False Acceptance Rate / False Rejection Rate)
