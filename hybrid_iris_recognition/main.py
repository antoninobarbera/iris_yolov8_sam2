import os
import sys
import shutil
import warnings
from pathlib import Path

from iris import iris_class
from tools.file_manager import *
from identification import id_class

import gc
import torch
import cv2
from tools.utils import iris_code_plot, plot_far_frr_vs_threshold

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def generate_folder(config):
    """
    Generates the necessary folders for storing iris images and related data.
    """   
    if os.path.exists(config.folders_path.iris_images):
        shutil.rmtree(config.folders_path.iris_images)
    
    os.makedirs(config.folders_path.iris_images, exist_ok=True)
    os.makedirs(config.folders_path.original_iris, exist_ok=True)
    os.makedirs(config.folders_path.segmented_iris, exist_ok=True)
    os.makedirs(config.folders_path.normalized_iris, exist_ok=True)
    os.makedirs(config.folders_path.iris_code, exist_ok=True)
    os.makedirs(config.folders_path.keypoints_iris, exist_ok=True)
    os.makedirs(config.folders_path.mask_sam2, exist_ok=True)
    os.makedirs(config.folders_path.mask_iris, exist_ok=True)
    os.makedirs(config.folders_path.mask_pupil, exist_ok=True)
   
   
def load_iris(eye, sub_index, image_index, yolo_model, sam_predictor, config, device):
    """
    Loads the iris from the eye image, performs segmentation, feature extraction, and sets the iris code.
    
    Returns:
        - iris_obtained: An instance of the iris_class containing the processed iris data.
    """
    iris_obtained = iris_class(eye, sub_index, config, device)
    
    if 'segmentation' in config.system_mode:
        iris_obtained.segmentation(yolo_model, sam_predictor, config)
        
        if config.graph.save_images:
            path = os.path.join('hybrid_iris_recognition','iris_images', 'original_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_image())
            path = os.path.join('hybrid_iris_recognition','iris_images', 'segmented_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_segmented_image())
        
    elif 'identification' in config.system_mode:
        iris_obtained.segmentation(yolo_model, sam_predictor, config)
        iris_obtained.feature_extraction()

        if 'daugman' in config.feature_extractor:
            iris_obtained.set_iris_code()   
            iris_code = iris_obtained.get_iris_code()
            
        if 'resnet101' in config.feature_extractor:
            iris_obtained.extract_resnet_features()
            iris_code = iris_obtained.get_resnet_features()
         
        if 'densenet201' in config.feature_extractor:
            iris_obtained.extract_densenet_features()
            iris_code = iris_obtained.get_densenet_features()
        
        if 'vit' in config.feature_extractor:
            iris_obtained.extract_vit_features()
            iris_code = iris_obtained.get_vit_features()
        
        if 'swin' in config.feature_extractor:
            iris_obtained.extract_swin_features()
            iris_code = iris_obtained.get_swin_features()
            
        if 'hybrid' in config.feature_extractor:
            iris_obtained.extract_hybrid_features()
            iris_code = iris_obtained.get_hybrid_features()
            
        if config.graph.save_images:
            path = os.path.join('hybrid_iris_recognition','iris_images', 'original_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_image())
            path = os.path.join('hybrid_iris_recognition','iris_images', 'segmented_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_segmented_image())
            path = os.path.join('hybrid_iris_recognition','iris_images', 'normalized_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_normalized_image())
            path = os.path.join('hybrid_iris_recognition','iris_images', 'iris_code', str(image_index)+'.jpeg')
            iris_code_plot(iris_code, path)
            path = os.path.join('hybrid_iris_recognition','iris_images', 'keypoints_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_keypoints_image())
            path = os.path.join('hybrid_iris_recognition', 'iris_images', 'mask_sam2', 'mask_iris', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_mask_iris())
            path = os.path.join('hybrid_iris_recognition', 'iris_images', 'mask_sam2', 'mask_pupil', str(image_index)+'.jpeg')
            cv2.imwrite(path, iris_obtained.get_mask_pupil())
         
    return iris_obtained


#### UBIRIS V1 dataset ####
def load_irises_ubirisv1(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the irises from the UBIRIS V1 dataset.

    Args:
        dataset (list): List of images to be processed.
        yolo_model: YOLO model for object detection.
        sam_predictor: SAM2 model for segmentation.
        config: Configuration object containing settings.
        device: Device to run the models on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - irises: List of iris objects for test images.
            - irises_stored: Dictionary of iris objects for training images, indexed by subject.
    """
    irises = []
    irises_stored = {i: [] for i in range(197)}
    index = 0

    for rec_index, rec in enumerate(dataset): 
        for subject in range(len(rec)):
            eye = dataset[rec_index][subject]

            if subject >= 189:
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)
            elif rec_index in [0, 1, 2]:
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises_stored[subject].append(iris_obtained)
            else:
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)

            index += 1

    return irises, irises_stored


#### CASIA V1 dataset ####
def load_irises_casiav1(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the irises from the CASIA V1 dataset.

    Args:
        dataset (list): List of images to be processed.
        yolo_model: YOLO model for object detection.
        sam_predictor: SAM2 model for segmentation.
        config: Configuration object containing settings.
        device: Device to run the models on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - irises: List of iris objects for test images.
            - irises_stored: Dictionary of iris objects for training images, indexed by subject.
    """
    irises = []
    irises_stored = {i: [] for i in range (0, 100)}
    index = 0

    for rec_index, rec in enumerate(dataset):
        for subject in range(0, 108):
            eye = rec[subject]
      
            if subject >= 100: # Unauthorized subject
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)
            elif rec_index in [0, 1, 4, 5]: # train samples
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises_stored[subject].append(iris_obtained)
            else: # test samples
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)
            index += 1

    return irises, irises_stored


#### CASIA V3 INTERVAL dataset ####
def load_irises_casiav3_interval(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the irises from the CASIA v3 interval dataset.

    Args:
        dataset (list): List of images to be processed.
        yolo_model: YOLO model for object detection.
        sam_predictor: SAM2 model for segmentation.
        config: Configuration object containing settings.
        device: Device to run the models on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - irises: List of iris objects for test images.
            - irises_stored: Dictionary of iris objects for training images, indexed by subject.
    """
    irises = []
    irises_stored = {i: [] for i in range(len(dataset[0]))}
    index = 0

    for rec_index, rec in enumerate(dataset):
        for subject in range(len(rec)):
            eye = dataset[rec_index][subject]

            if rec_index in [0, 2, 4]:  # train
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises_stored[subject].append(iris_obtained)
            else:  # test
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)

            index += 1

    return irises, irises_stored


#### CASIA V3 LAMP dataset ####
def load_irises_casiav3_lamp(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the irises from the CASIA v3 lamp dataset.

    Args:
        dataset (list): List of images to be processed.
        yolo_model: YOLO model for object detection.
        sam_predictor: SAM2 model for segmentation.
        config: Configuration object containing settings.
        device: Device to run the models on (CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - irises: List of iris objects for test images.
            - irises_stored: Dictionary of iris objects for training images, indexed by subject.
    """
    irises = []
    irises_stored = {i: [] for i in range(len(dataset[0]))}
    index = 0

    for rec_index, rec in enumerate(dataset):
        for subject in range(len(rec)): 
            eye = dataset[rec_index][subject]

            #if rec_index in [0, 2, 3, 5, 8, 10, 11, 12, 14, 15, 17, 18, 19]:
            if rec_index in [0, 3, 5, 10, 12, 15, 18]:
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises_stored[subject].append(iris_obtained)
            else:
                iris_obtained = load_iris(eye, subject, index, yolo_model, sam_predictor, config, device)
                if iris_obtained is None:
                    index += 1
                    continue
                irises.append(iris_obtained)

            index += 1

    return irises, irises_stored


#### Load segmentation dataset ####
def load_segmentation_test1(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the segmentation test dataset and performs segmentation on each image.
    
    Args:
        - dataset: List of images to be segmented.
        - yolo_model: YOLO model for object detection.
        - sam_predictor: SAM2 model for segmentation.
        - config: Configuration object containing settings.
        - device: Device to run the models on (CPU or GPU).
    """
    segmented_count = 0
    failed_count = 0

    for index, img in enumerate(dataset):
        iris_obtained = load_iris(img, index, index, yolo_model, sam_predictor, config, device)

        if iris_obtained is None:
            failed_count += 1
            continue

        segmented_count += 1

    print(f"\nSegmentation completed:")
    print(f"   - Total images: {len(dataset)}")
    print(f"   - Successfully segmented: {segmented_count}")
    print(f"   - Failed: {failed_count}")
    
    
def load_segmentation_test(dataset, yolo_model, sam_predictor, config, device):
    """
    Loads the segmentation test dataset and performs segmentation on each image.

    Args:
        - dataset: List of images to be segmented.
    """
    print(f"\nDataset length: {len(dataset)}")

    segmented_count = 0
    failed_count = 0

    for index, img in enumerate(dataset):
        try:
            iris_obtained = load_iris(img, index, index, yolo_model, sam_predictor, config, device)
        except Exception as e:
            print(f"[Errore] Immagine {index}: {e}")
            failed_count += 1
            continue

        if iris_obtained is None:
            failed_count += 1
        else:
            segmented_count += 1

    print(f"\nSegmentation completed:")
    print(f"   - Total images: {len(dataset)}")
    print(f"   - Successfully segmented: {segmented_count}")
    print(f"   - Failed: {failed_count}")


#### Test function for identification ####
def test(irises, irises_stored, threshold=None):
    """
    Tests the identification system using the provided irises and stored irises

    Args:
        irises (list): List of iris objects to be tested.
        irises_stored (dict): Dictionary of stored iris objects indexed by subject.
        threshold (int, optional): Threshold value for identification. If None, uses default threshold.

    Returns:
        tuple: FAR (False Acceptance Rate) and FRR (False Rejection Rate) percentages.
    """
    if threshold is not None:
        print(" Testing Threshold " + str(threshold) + ':')
    id = id_class(config, irises_stored)
    tp, fp, tn, fn, tot = 0, 0, 0, 0, 0

    for iris in irises:
        tot += 1
        
        if 'hybrid' in config.feature_extractor:
            flag, label = id.identification_hybrid(iris, threshold)
        else:
            flag, label, = id.identification(iris, threshold)

        # Evaluate identification results
        if flag:
            if iris.get_idx() == label:
                tp += 1
            else:
                fp += 1
        else:
            if iris.get_idx() < 100:
               fn += 1
            else:
               tn += 1
   
    accuracy = (tp + tn) / tot * 100
    far = fp / (fp + tn) * 100
    frr = fn / (fn + tp) * 100
   
    print('\n Performance achieved:')
    print('\taccuracy ' + str(round(accuracy, 2)) + " %")
    print('\tFAR ' + str(round(far, 2)) + " %")
    print('\tFRR ' + str(round(frr, 2)) + " %")
    print('-----------------------------------------------------------------\n')

    return far, frr

        

if  __name__ == '__main__':

    # Load the configuration file
    config = configuration()
    
    #### LOADING CONFIGURATION ###    
    sam2_type = config.sam2_type 
    checkpoint_sam2 = getattr(config.checkpoints_sam2, sam2_type)
    model_cfg = getattr(config.config_sam2, sam2_type)
    checkpoint_yolov8n = config.checkpoints_yolov8.yolov8_model
    
    #### LOADING MODELS ####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
    
    yolo_model = YOLO(checkpoint_yolov8n, verbose=False)
    sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_sam2).to(device))
    
    print("\n------------------------------------------------------------------------------------------------------------------")
    print('Start SIFT + Classifiers system')
    print(f"\nLoading models:\n  - YOLOv8 model: {Path(checkpoint_yolov8n).name}\n  - SAM2 model: {Path(checkpoint_sam2).name}\n")
    
    generate_folder(config)
    
    if 'identification' in config.system_mode:
        print("System mode: Identification")
        
        if 'casiav1' in config.dataset_to_use:
            print("Loading CASIA v1 dataset...")
            dataset = load_dataset_casiav1(config)
            print('\n Original - Segmentation - Normalization - Iris Code - Keypoints - Images Generation in progress...')
            irises, irises_stored = load_irises_casiav1(dataset, yolo_model, sam_predictor, config, device)
        
        elif 'ubirisv1' in config.dataset_to_use:
            print("Loading Ubiris v1 dataset...")
            dataset = load_dataset_ubirisv1(config)
            print('\n Original - Segmentation - Normalization - Iris Code - Keypoints - Images Generation in progress...')
            irises, irises_stored = load_irises_ubirisv1(dataset, yolo_model, sam_predictor, config, device)
               
        elif 'casia_v3_interval' in config.dataset_to_use:
            print("Loading Casia-v3-Interval dataset...")
            dataset = load_dataset_casia_interval(config)
            print('\n Original - Segmentation - Normalization - Iris Code - Keypoints - Images Generation in progress...')
            irises, irises_stored = load_irises_casiav3_interval(dataset, yolo_model, sam_predictor, config, device)
            
        elif 'casia_v3_lamp' in config.dataset_to_use:
            print("Loading Casia-v3-Lamp dataset...")
            dataset = load_dataset_casia_lamp(config)
            dataset = [dataset[0], dataset[3], dataset[5], dataset[10], dataset[12], dataset[15], dataset[18], dataset[1], dataset[4], dataset[13], dataset[16]] 
            print('\n Original - Segmentation - Normalization - Iris Code - Keypoints - Images Generation in progress...')
            irises, irises_stored = load_irises_casiav3_lamp(dataset, yolo_model, sam_predictor, config, device)
            
        else:
            print('Dataset not recognized. Please check the configuration file.')
            sys.exit(1)
        
        # Test system
        thresholds = config.test_thresholds
        far = []
        frr = []

        print('-----------------------------------------------------------------')
        
        for threshold in thresholds:
                far_x, frr_x = test(irises, irises_stored, threshold)
                far.append(far_x)
                frr.append(frr_x)

        if not directory_exists('hybrid_iris_recognition/graph'):
            os.mkdir('hybrid_iris_recognition/graph')
        
        path = os.path.join('hybrid_iris_recognition/graph', 'roc.png')   
        plot_far_frr_vs_threshold(far, frr, thresholds, path)
    
    elif 'segmentation' in config.system_mode:
        print("System mode: Segmentation")
        
        if 'casiav1' in config.dataset_to_use:
            print("Loading CASIA v1 dataset for segmentation test...")
            dataset = load_casiav1_flat(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
        
        elif 'ubirisv1' in config.dataset_to_use:
            print("Loading Ubiris v1 dataset for segmentation test...")
            dataset = load_ubirisv1_flat(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
            
        elif 'ubirisv2' in config.dataset_to_use:
            print("Loading Ubiris v1 dataset for segmentation test...")
            dataset = load_dataset_sessao2(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
            
        elif 'casia_v3_interval' in config.dataset_to_use:
            print("Loading Casia-v3-Interval dataset...")
            dataset = load_casia_interval_flat(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
            
        elif 'casia_v3_lamp' in config.dataset_to_use:
            print("Loading Casia-v3-Lamp dataset...")
            dataset = load_casia_lamp_flat(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
            
        elif 'casia_v3_twins' in config.dataset_to_use:
            print("Loading Casia-v3-Lamp dataset...")
            dataset = load_dataset_casia3_twins(config)
            print('\n Original - Segmentation - Images Generation in progress...')
            load_segmentation_test(dataset, yolo_model, sam_predictor, config, device)
            
        else:
            print('Dataset not recognized. Please check the configuration file.')
            sys.exit(1)
    else:
        print('System mode not recognized. Please check the configuration file.')
        sys.exit(1)
        
    print("\nEnd")
    