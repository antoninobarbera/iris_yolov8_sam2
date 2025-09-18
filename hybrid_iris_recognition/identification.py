import os
import sys
import joblib
import cv2 as cv
from tools.matching_score import matching_score_class

class id_class:
    """
    A class to handle iris identification and matching operations.
    """
    __slots__ = ['config', 'are_models_loaded','scaler', 'feature_reduction', 'classifier_1', 'classifier_2', 'classifier_3', 'data_dict']

    def __init__(self, config, data_dict=None):
        """
        Initializes the id_class object.

        Parameters:
        - config: Configuration object containing settings and parameters.
        - data_dict: Optional dictionary of preloaded iris data for identification.
        """
        self.config = config
        if data_dict is not None:
           self.are_models_loaded = True
           scaler_path = os.path.join('hybrid_iris_recognition/checkpoints', 'standard_scaler.pkl')
           self.scaler = joblib.load(scaler_path)
           feature_reduction_path = os.path.join('hybrid_iris_recognition/checkpoints', 'lle_feature_reduction.pkl')
           self.feature_reduction = joblib.load(feature_reduction_path)
           classifier_1_path = os.path.join('hybrid_iris_recognition/checkpoints', 'knn_model.pkl')
           self.classifier_1 = joblib.load(classifier_1_path)
           classifier_2_path = os.path.join('hybrid_iris_recognition/checkpoints', 'svm_model.pkl')
           self.classifier_2 = joblib.load(classifier_2_path)
           classifier_3_path = os.path.join('hybrid_iris_recognition/checkpoints', 'nn_model.pkl')
           self.classifier_3 = joblib.load(classifier_3_path)        
           self.data_dict = data_dict
        else:
            self.are_models_loaded = False

    
    def sift_match(self, iris_1, iris_2, threshold=None):
        """
        Performs SIFT-based matching between two iris samples.

        Parameters:
        - iris_1: First iris object.
        - iris_2: Second iris object.
        - threshold: Optional matching score threshold.

        Returns:
        - flag: Boolean indicating whether the irises match.
        """
        bf = cv.BFMatcher()
        kp_1 = iris_1.get_keypoints()
        des_1 = iris_1.get_descriptors()
        kp_2 = iris_2.get_keypoints()
        des_2 = iris_2.get_descriptors()
        if not kp_1 or not kp_2: 
            return False
        matches = bf.knnMatch(des_1, des_2, k=2)
        matching_score = matching_score_class(iris_1, iris_2, self.config)
        for m, n in matches:
            if (m.distance / n.distance) > self.config.matching.lowe_filter:
                continue
            x_1, y_1 = kp_1[m.queryIdx].pt
            x_2, y_2 = kp_2[m.trainIdx].pt
            p_1 = (x_1, y_1)
            p_2 = (x_2, y_2) 
            matching_score.__add__(p_1, p_2)
        score = matching_score()
        if threshold is None:
            threshold = self.config.matching.threshold
        if score > threshold:
            flag = True
        else:
            flag = False
        return flag
    
    
    def identification(self, iris, threshold):
        """
        Identifies an iris by comparing it to known data.

        Parameters:
        - iris: The iris object to identify.
        - threshold: Matching threshold to determine a match.

        Returns:
        - result: A tuple (is_match, label).
            - is_match: True if a match is found, False otherwise.
            - label: The identifier of the matched iris, or None if no match.
        """
        if not self.are_models_loaded:
            print(' MODELS ARE NOT UPLOADED --- RUN models_generation.py')
            sys.exit()
        iris_code = [iris.get_iris_code()]
        iris_code_scaled = self.scaler.transform(iris_code)
        iris_code_red = self.feature_reduction.transform(iris_code_scaled)
        
        # CLASSIFIER 1 MATCHING
        label_1 = self.classifier_1.predict(iris_code_red)       
        # CLASSIFIER 2 MATCHING
        label_2 = self.classifier_2.predict(iris_code_red)      
        # CLASSIFIER 3 MATCHING
        label_3 = self.classifier_3.predict(iris_code_red)

        possible = [label_1[0], label_2[0], label_3[0]]
        possible = list(set(possible))

        result = []
        for label in possible:
            
            if label not in self.data_dict:
                continue 
    
            for possible_iris in self.data_dict[label]:
                if self.sift_match(iris, possible_iris, threshold):
                    result.append(label)
                    break
        result = list(set(result))
        
        if len(result) == 1:
            return True, result[0]
        else:
            return False, None
    
    
    def identification_hybrid(self, iris, threshold):
        """
        Identifies an iris by comparing it to known data.

        Parameters:
        - iris: The iris object to identify.
        - threshold: Matching threshold to determine a match.

        Returns:
        - result: A tuple (is_match, label).
            - is_match: True if a match is found, False otherwise.
            - label: The identifier of the matched iris, or None if no match.
        """
        if not self.are_models_loaded:
            print(' MODELS ARE NOT UPLOADED --- RUN models_generation.py')
            sys.exit()
        iris_code = [iris.get_hybrid_features()]
        iris_code_scaled = self.scaler.transform(iris_code)
        iris_code_red = self.feature_reduction.transform(iris_code_scaled)
        
        # CLASSIFIER 1 MATCHING
        label_1 = self.classifier_1.predict(iris_code_red)       
        # CLASSIFIER 2 MATCHING
        label_2 = self.classifier_2.predict(iris_code_red)      
        # CLASSIFIER 3 MATCHING
        label_3 = self.classifier_3.predict(iris_code_red)

        possible = [label_1[0], label_2[0], label_3[0]]
        possible = list(set(possible))

        result = []
        for label in possible:
            
            if label not in self.data_dict:
                continue 
    
            for possible_iris in self.data_dict[label]:
                if self.sift_match(iris, possible_iris, threshold):
                    result.append(label)
                    break
        result = list(set(result))

        if len(result) == 1:
            return True, result[0]
        else:
            return False, None
        