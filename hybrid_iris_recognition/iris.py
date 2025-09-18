import os
import sys
import cv2
import numpy as np
from PIL import Image
from tools.utils import point_in_circle, draw_keypoints_image, warp_polar, encoding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_iris_recognition.inference import yolo_sam_inference

import torch
from hybrid_iris_recognition.feature_extractor.ResNet101.resnet101_model import get_resnet_model, get_resnet_preprocess
from feature_extractor.ViT.vit_model import get_vit_model, get_vit_preprocess
from feature_extractor.Swin_transformer.swin_model import get_swin_model, get_swin_preprocess
from feature_extractor.DenseNet.densenet import get_densenet_model, get_densenet_preprocess


class iris_class:
    
    __slots__ = ['image', 'config', 'centre', 'idx',
                 'pupil_radius', 'iris_radius', 'segmented', 'mask_iris',
                 'mask_pupil', 'keypoints', 'descriptors', 'keypoints_image','features',
                 'preprocess', 'normalize', 'iris_code', 'resnet_model', 'resnet_preprocess',
                 'resnet_features', 'vit_model', 'swin_preprocess', 'swin_model', 'vit_preprocess',
                 'vit_features', 'swin_features', 'densenet_features', 'densenet_model',
                 'densenet_preprocess', 'hybrid_features', 'device']


    def __init__(self, image, idx, config, device):
        
        self.image = image
        self.config = config
        self.idx = idx
        self.device = device
        
        if 'resnet101' in config.feature_extractor:
            self.resnet_model = get_resnet_model(self.device)
            self.resnet_preprocess = get_resnet_preprocess()
        if 'vit' in config.feature_extractor:
            self.vit_model = get_vit_model(self.device)
            self.vit_preprocess = get_vit_preprocess()
        if 'swin' in config.feature_extractor:
            self.swin_model = get_swin_model(self.device)
            self.swin_preprocess = get_swin_preprocess()
        if 'densenet201' in config.feature_extractor:
            self.densenet_model = get_densenet_model(self.device)
            self.densenet_preprocess = get_densenet_preprocess()
        if 'hybrid' in config.feature_extractor:
            self.densenet_model = get_densenet_model(self.device)
            self.densenet_preprocess = get_densenet_preprocess()
            #self.swin_model = get_swin_model(self.device)
            #self.swin_preprocess = get_swin_preprocess()


    def segmentation(self, yolo_model, sam_predictor, config):
        """
        Segments the iris region from the input image.

        Steps:
        1. Calls the `yolo_sam_inference` function to perform segmentation using YOLO and SAM.
        2. Extracts the centre, pupil radius, and iris radius from the segmentation result. 
        3. Saves the segmented iris image in the `self.segmented` attribute.
        
        Result:
        - The segmented iris is saved in the `self.segmented` attribute.
        """
        segmented, centre, pupil_radius, iris_radius, mask_iris, mask_pupil = yolo_sam_inference(self.image, yolo_model, sam_predictor, config, self.idx)
        self.centre = centre
        self.pupil_radius = pupil_radius
        self.iris_radius = iris_radius
        self.segmented = segmented
        self.mask_iris = mask_iris
        self.mask_pupil = mask_pupil
            
            
    def normalization(self):
        """
        Normalizes the segmented iris region into polar coordinates.

        Steps:
        1. Warps the segmented iris image into polar coordinates.
        2. Inverts pixel values and converts the result to an 8-bit image.
        3. Crops the normalized image using the configuration border size.
        
        Result:
        - The normalized iris image is saved in the `self.normalize` attribute.
        """
        image = self.get_segmented_image()
        x_size = self.config.normalization.x_size
        y_size = self.config.normalization.y_size        
        polar_iris = warp_polar(image, x_size, y_size, self.centre, self.pupil_radius, self.iris_radius)
        polar_iris_inv = 255 - polar_iris
        polar_iris_uint8 = polar_iris_inv.astype(np.uint8)
        normalized_iris = polar_iris_uint8[0:self.config.normalization.border, :]
        self.normalize = normalized_iris
         
         
    def feature_extraction(self):
        """
        Extracts keypoints and descriptors from the segmented iris using SIFT.

        Steps:
        1. Detects keypoints using SIFT.
        2. Filters valid keypoints based on their position:
           - Inside the iris region.
           - Outside the pupil region.
        3. Computes descriptors for the valid keypoints.
        4. Draws keypoints on the segmented image for visualization.

        Results:
        - The keypoints are saved in `self.keypoints`.
        - The descriptors are saved in `self.descriptors`.
        - The visualization image is saved in `self.keypoints_image`.
        """
        sift = cv2.SIFT_create()
        kp_found = sift.detect(self.segmented, None)
        valid_kp = []
        for kp in kp_found:
            point = (kp.pt[0], kp.pt[1])
            kp_in_iris = point_in_circle(self.centre, self.iris_radius, point)
            kp_out_pupil = not point_in_circle(self.centre, self.pupil_radius, point)
            if kp_in_iris and kp_out_pupil:
                valid_kp.append(kp)
        self.keypoints, self.descriptors = sift.compute(self.segmented, valid_kp)
        self.keypoints_image = draw_keypoints_image(self.segmented, valid_kp, self.centre, self.iris_radius) 
         

    def set_iris_code(self):
        """
        Encodes the normalized iris into a binary code.

        Steps:
        1. Calls the `normalization` method to ensure the iris is normalized.
        2. Uses the `encoding` function to generate a binary iris code.

        Result:
        - The binary iris code is saved in the `self.iris_code` attribute.
        """
        self.normalization()
        self.iris_code = encoding(self.normalize, self.config)
        
        
    def extract_resnet_features(self):
        """
        Extracts features using ResNet from the normalized iris image.
        """
        self.normalization()

        img_resized = cv2.resize(self.normalize, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        input_tensor = self.resnet_preprocess(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet_model(input_tensor)

        self.resnet_features = features.squeeze().cpu().numpy()
        
        
    def extract_vit_features(self):
        """
        Extracts features using Vision Transformer (ViT) from the normalized iris image.
        """
        self.normalization()

        img_resized = cv2.resize(self.normalize, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        input_tensor = self.vit_preprocess(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.vit_model(input_tensor)

        self.vit_features = features.squeeze().cpu().numpy()
    
    
    def extract_swin_features(self):
        """
        Extracts features using Swin Transformer from the normalized iris image.
        """
        self.normalization()

        img_resized = cv2.resize(self.normalize, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)

        input_tensor = self.swin_preprocess(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.swin_model(input_tensor)

        self.swin_features = features.squeeze().cpu().numpy()

    
    def extract_densenet_features(self):
        """
        Extracts features using DenseNet from the normalized iris image.
        """
        self.normalization()
        
        img_resized = cv2.resize(self.normalize, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.densenet_preprocess(img_pil)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.densenet_model(input_tensor)

        features = features.squeeze().cpu().numpy()
        self.densenet_features = features
    
    
    def extract_hybrid_features(self):
        """
        Extracts hybrid features by combining the iris code and features from the Swin Transformer.
        """
        self.normalization()
        self.iris_code = encoding(self.normalize, self.config)
        
        if len(self.segmented.shape) == 2:
            img_rgb = cv2.cvtColor(self.segmented, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = self.segmented

        # densenet
        img_pil_densenet = Image.fromarray(img_rgb)
        input_densenet = self.densenet_preprocess(img_pil_densenet).unsqueeze(0).to(self.device)
        with torch.no_grad():
            resnet_feat = self.densenet_model(input_densenet)  # [1, 2048, 1, 1]
        resnet_feat = resnet_feat.view(-1).cpu().numpy()  # [2048]

        self.hybrid_features = np.concatenate([self.iris_code, resnet_feat])
        
        
    def extract_hybrid_features1(self):
        """
        Extracts hybrid features from the normalized iris image using both DenseNet and Swin Transformer.
        """
        self.normalization()
        self.iris_code = encoding(self.normalize, self.config)
        
        if len(self.segmented.shape) == 2:
            img_rgb = cv2.cvtColor(self.segmented, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = self.segmented

        # Swin Transformer
        img_pil = Image.fromarray(img_rgb)
        input_swin = self.swin_preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            swin_feat = self.swin_model(input_swin)  # [1, 768]
        swin_feat = swin_feat.view(-1).cpu().numpy()  # [768]

        self.hybrid_features = np.concatenate([self.iris_code, swin_feat])


    def get_idx(self):
        """
        Returns the index or identifier of the iris sample.
        """
        return self.idx
    
    def get_image(self):
        """
        Returns the original input image.
        """
        return self.image
 
    def get_iris_code(self):
        """
        Returns the encoded binary iris code.
        """
        return self.iris_code
    
    def get_keypoints(self):
        """
        Returns the detected keypoints from the segmented iris.
        """
        return self.keypoints
    
    def get_descriptors(self):
        """
        Returns the descriptors computed for the valid keypoints.
        """
        return self.descriptors
    
    def get_mask_iris(self):
        """
        Returns the iris mask.
        """
        return self.mask_iris
    
    def get_mask_pupil(self):
        """
        Returns the pupil mask.
        """
        return self.mask_pupil

    
    def get_segmented_image(self):
        """
        Returns the segmented iris image.
        """
        return self.segmented
    
    def get_keypoints_image(self):
        """
        Returns the image with keypoints drawn for visualization.
        """
        return self.keypoints_image
    
    def get_normalized_image(self):
        """
        Returns the normalized iris image in polar coordinates.
        """
        return self.normalize
    
    def get_attributes(self):
        """
        Returns key attributes of the iris:
        - Centre of the iris.
        - Pupil radius.
        - Iris radius.
        """
        return self.centre, self.pupil_radius, self.iris_radius

    def get_resnet_features(self):
        """
        Returns the features extracted using ResNet.
        """
        return getattr(self, 'resnet_features', None)
    
    def get_vit_features(self):
        """
        Returns the features extracted using vit.
        """
        return getattr(self, 'vit_features', None)

    def get_swin_features(self):
        """
        Returns the features extracted using swin.
        """
        return getattr(self, 'swin_features', None)
    
    def get_densenet_features(self):
        """
        Returns the features extracted using densenet.
        """
        return getattr(self, 'densenet_features', None)
    
    def get_hybrid_features(self):
        """
        Returns the features extracted using hybrid approach.
        """
        return getattr(self, 'hybrid_features', None)
    