import os
import gc
import sys
from pathlib import Path

import torch
import joblib
import numpy as np
from tqdm import tqdm
from iris import iris_class
from nn_model.nn_iris import iris_network
from tools.file_manager import *

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
from nn_model.nn_iris_model import nn_classifier_class

from tools.utils import accuracy_comparison, LLE_graph


def create_train_test():
   """
   Prepares training and testing datasets for model training and evaluation.
   
   Returns:
      tuple: A tuple containing:
         - train (list): List of training datasets.
         - test (list): List of testing datasets.
         - yolo_model (YOLO): YOLO model for object detection.
         - sam_predictor (SAM2ImagePredictor): SAM2 image predictor for segmentation.
         - device (str): Device type ('cuda' or 'cpu').
   """
   if 'resnet101' in config.feature_extractor:
      print("Feature extractor: ResNet101")
   elif 'densenet201' in config.feature_extractor:
      print("Feature extractor: Densenet201")
   elif 'swin' in config.feature_extractor:
      print("Feature extractor: Swin transformer")
   elif 'daugman' in config.feature_extractor:
      print("Feature extractor: Gabor filter (Daugman)")
   elif 'vit' in config.feature_extractor:
      print("Feature extractor: ViT")
   elif 'hybrid' in config.feature_extractor:
      print("Feature extractor: Daugman + Densenet201")
   else:
      print("Feature extractor: Unknown or not specified")
      
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

   #### LOADING DATASET ###
   if 'casiav1' in config.dataset_to_use:
      print("Loading CASIA v1 dataset...")
      dataset = load_dataset_casiav1(config)
      train = [dataset[0], dataset[1] , dataset[4], dataset[5]]
      test = [dataset[2], dataset[3] , dataset[6]]
      
   elif 'ubirisv1' in config.dataset_to_use:
      print("Loading Ubiris v1 dataset...")
      dataset = load_dataset_ubirisv1(config)
      train = [dataset[0], dataset[1], dataset[2]]
      test = [dataset[3], dataset[4]]
      
   elif 'casia_v3_interval' in config.dataset_to_use:
      print("Loading Casia v3 interval dataset...")
      dataset = load_dataset_casia_interval(config)
      train = [dataset[0], dataset[2], dataset[4]]
      test = [dataset[1], dataset[3]]
      
   elif 'casia_v3_lamp' in config.dataset_to_use:
      print("Loading Casia v3 lamp dataset...")
      #create_casia_lamp_pickle()
      dataset = load_dataset_casia_lamp(config)
      train = [dataset[0], dataset[2], dataset[3], dataset[5], dataset[8], dataset[10], dataset[11], dataset[12], dataset[14], dataset[15], dataset[17], dataset[18], dataset[19]]
      #train = [dataset[0], dataset[3], dataset[5], dataset[10], dataset[12], dataset[15], dataset[18]]
      test = [dataset[1], dataset[4], dataset[6], dataset[7], dataset[9], dataset[13], dataset[16]]
      #test = [dataset[1], dataset[4], dataset[13], dataset[16]]
      
   elif config.dataset_to_use == 'all':
      print("Loading all datasets...")

      dataset_casia_v1 = load_dataset_casiav1(config)
      dataset_ubiris = load_dataset_ubirisv1(config)
      dataset_casia_interval = load_dataset_casia_interval(config)
      dataset_casia_lamp = load_dataset_casia_lamp(config)

      train = [
         dataset_casia_v1[0], dataset_casia_v1[1], dataset_casia_v1[4], dataset_casia_v1[5],
         dataset_ubiris[0], dataset_ubiris[1], dataset_ubiris[2],
         dataset_casia_interval[0], dataset_casia_interval[2], dataset_casia_interval[4],
         dataset_casia_lamp[0], dataset_casia_lamp[2], dataset_casia_lamp[3], dataset_casia_lamp[5],
         dataset_casia_lamp[8], dataset_casia_lamp[10], dataset_casia_lamp[11], dataset_casia_lamp[12],
         dataset_casia_lamp[14], dataset_casia_lamp[15], dataset_casia_lamp[17], dataset_casia_lamp[18],
         dataset_casia_lamp[19]
      ]

      test = [
         dataset_casia_v1[2], dataset_casia_v1[3], dataset_casia_v1[6],
         dataset_ubiris[3], dataset_ubiris[4],
         dataset_casia_interval[1], dataset_casia_interval[3],
         dataset_casia_lamp[1], dataset_casia_lamp[4], dataset_casia_lamp[6], dataset_casia_lamp[7],
         dataset_casia_lamp[9], dataset_casia_lamp[13], dataset_casia_lamp[16]
      ]

   else:
      print('Dataset not recognized. Please check the configuration file.')
      sys.exit(1)
      
   print(f"\nLoading models:\n  - YOLOv8 model: {Path(checkpoint_yolov8n).name}\n  - SAM2 model: {Path(checkpoint_sam2).name}")
   print('\nCreating train set and test set...\n')
   
   return train, test, yolo_model, sam_predictor, device


def train_test():
   """
   Prepares training and testing datasets for model training and evaluation.

   :return: Scaled and formatted training and testing data with corresponding labels.
   :rtype: tuple (numpy.ndarray, list, numpy.ndarray, list)
   """
   train, test, yolo_model, sam_predictor, device = create_train_test()
   
   #### LOADING TRAIN SET ###
   X_train_temp = []
   y_train = []
   
   for i, rec in enumerate(train):
      for j, eye in enumerate(tqdm(rec, desc="Training progress", total=len(rec)*len(train))):
         iris_obtained = iris_class(eye, None, config, device)
         iris_obtained.segmentation(yolo_model, sam_predictor, config)
         
         if 'daugman' in config.feature_extractor:
            iris_obtained.set_iris_code()   
            features = iris_obtained.get_iris_code()
          
         if 'resnet101' in config.feature_extractor:
            iris_obtained.extract_resnet_features()
            features = iris_obtained.get_resnet_features()
         
         if 'densenet201' in config.feature_extractor:
            iris_obtained.extract_densenet_features()
            features = iris_obtained.get_densenet_features()
            
         if 'vit' in config.feature_extractor:
            iris_obtained.extract_vit_features()
            features = iris_obtained.get_vit_features()
            
         if 'swin' in config.feature_extractor:
            iris_obtained.extract_swin_features()
            features = iris_obtained.get_swin_features()
         
         if 'hybrid' in config.feature_extractor:
            iris_obtained.extract_hybrid_features()
            features = iris_obtained.get_hybrid_features()

         X_train_temp.append(features)
         y_train.append(j)
      print("\n")
   
   #### LOADING TEST SET ###
   X_test_temp = []
   y_test = []
   
   for i, rec in enumerate(test):
      for j, eye in enumerate(tqdm(rec, desc="Testing progress", total=len(rec)*len(test))):
         iris_obtained = iris_class(eye, None, config, device)
         iris_obtained.segmentation(yolo_model, sam_predictor, config)
         
         if 'daugman' in config.feature_extractor:
            iris_obtained.set_iris_code()   
            features = iris_obtained.get_iris_code()
            
         if 'resnet101' in config.feature_extractor:
            iris_obtained.extract_resnet_features()
            features = iris_obtained.get_resnet_features()
         
         if 'densenet201' in config.feature_extractor:
            iris_obtained.extract_densenet_features()
            features = iris_obtained.get_densenet_features()
            
         if 'vit' in config.feature_extractor:
            iris_obtained.extract_vit_features()
            features = iris_obtained.get_vit_features()
            
         if 'swin' in config.feature_extractor:
            iris_obtained.extract_swin_features()
            features = iris_obtained.get_swin_features()
            
         if 'hybrid' in config.feature_extractor:
            iris_obtained.extract_hybrid_features()
            features = iris_obtained.get_hybrid_features()
         
         X_test_temp.append(features)
         y_test.append(j)
      print("\n")

   X_train = np.vstack(X_train_temp)
   X_test = np.vstack(X_test_temp)

   return X_train, y_train, X_test, y_test, config.feature_extractor



if  __name__ == '__main__':
   
   config = configuration()
   
   print("\n------------------------------------------------------------------------------------------------------------------")
   print('Training start\n')
   
   X_train, y_train, X_test, y_test, model_name = train_test()

   # Standardize the datasets
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Apply feature reduction using Locally Linear Embedding (LLE)
   n_neighbors = config.feature_reduction_lle.n_neighbors
   n_components = config.feature_reduction_lle.n_components
   lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
   lle.fit(X_train_scaled)
   X_train_red = lle.transform(X_train_scaled)
   X_test_red = lle.transform(X_test_scaled)
   LLE_graph(X_train_red, y_train)

   # training knn
   print('  - Training KNN...')
   knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
   knn.fit(X_train_red, y_train)
   y_pred_knn = knn.predict(X_test_red)
   y_train_pred_knn = knn.predict(X_train_red)

   # training svm
   print('\n  - Training SVM...')
   svm = SVC(kernel='sigmoid')
   svm.fit(X_train_red, y_train)
   y_pred_svm = svm.predict(X_test_red)
   y_train_pred_svm = svm.predict(X_train_red)

   # training neural network
   print('\n  - Training Neural Network...')
   input_size = X_train_red.shape[1]
 
   num_classes = len(set(y_train))

   model = iris_network(input_size, num_classes)
   nn = nn_classifier_class(model, config)
   nn.fit(X_train_red, y_train)
   y_pred_nn = nn.predict(X_test_red)
   y_train_pred_nn = nn.predict(X_train_red)

   # Calculate test accuracy for each model
   accuracy_knn = accuracy_score(y_test, y_pred_knn)
   accuracy_svm = accuracy_score(y_test, y_pred_svm)
   accuracy_nn = accuracy_score(y_test, y_pred_nn)
   
   # Calculate merged accuracy for ensemble-like evaluation
   print('\nCalculating performance...')
   matched = 0
   for i in range(len(y_test)):
      if y_test[i] == y_pred_knn[i] or y_test[i] == y_pred_svm[i] or y_test[i] == y_pred_nn[i]:
         matched += 1

   merge_accuracy = matched / len(y_test)
   
   #accuracy_comparison(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy)
   accuracy_comparison(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy, model_name)

   #### Calculate train accuracy for each model
   accuracy_train_knn = accuracy_score(y_train, y_train_pred_knn)
   accuracy_train_svm = accuracy_score(y_train, y_train_pred_svm)
   accuracy_train_nn = accuracy_score(y_train, y_train_pred_nn)
   
   print("\nTRAIN performance...")
   print('    Accuracy KNN : ' + str(round(accuracy_train_knn * 100, 2)))
   print('    Accuracy SVM : ' + str(round(accuracy_train_svm * 100, 2)))
   print('    Accuracy NN : ' + str(round(accuracy_train_nn * 100, 2)))

   print("\nTEST performance...")
   print('    Accuracy KNN : ' + str(round(accuracy_knn * 100, 2)))
   print('    Accuracy SVM : ' + str(round(accuracy_svm * 100, 2)))
   print('    Accuracy NN : ' + str(round(accuracy_nn * 100, 2))) 
   print('    Merge Accuracy : ' + str(round(merge_accuracy * 100, 2)))
   print('\n')
   
   # Save the trained models and configurations
   if not directory_exists('hybrid_iris_recognition/checkpoints'):
      os.mkdir('hybrid_iris_recognition/checkpoints')
   
   joblib.dump(scaler, os.path.join('hybrid_iris_recognition/checkpoints', 'standard_scaler.pkl'))
   joblib.dump(lle, os.path.join('hybrid_iris_recognition/checkpoints', 'lle_feature_reduction.pkl'))
   joblib.dump(knn, os.path.join('hybrid_iris_recognition/checkpoints', 'knn_model.pkl'))
   joblib.dump(svm, os.path.join('hybrid_iris_recognition/checkpoints', 'svm_model.pkl'))
   joblib.dump(nn, os.path.join('hybrid_iris_recognition/checkpoints', 'nn_model.pkl'))
   