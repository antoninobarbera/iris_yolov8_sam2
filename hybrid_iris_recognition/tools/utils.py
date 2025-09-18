import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def distance(point_1, point_2):
    """
    Computes the Euclidean distance between two points in 2D space.

    Parameters:
    - point_1 (tuple): Coordinates (x, y) of the first point.
    - point_2 (tuple): Coordinates (x, y) of the second point.

    Returns:
    - dist (float): The Euclidean distance between the two points.
    """
    x1, y1 = point_1
    x2, y2 = point_2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def point_in_circle(centre, radius, point):
    """
    Checks if a point lies within a circle.

    Parameters:
    - centre (tuple): Coordinates (x, y) of the circle's center.
    - radius (float): The radius of the circle.
    - point (tuple): Coordinates (x, y) of the point to check.

    Returns:
    - bool: True if the point is within the circle, False otherwise.
    """
    dist = distance(centre, point) 
    return dist <= radius

def draw_keypoints_image(image_iris, keypoints, centre, iris_radius):
    """
    Draws keypoints and circles (for iris segmentation) on an image.

    Parameters:
    - image_iris (numpy.ndarray): The iris image to draw on.
    - keypoints (list): List of keypoints to draw.
    - centre (tuple): Coordinates (x, y) of the iris center.
    - iris_radius (int): Radius of the iris.

    Returns:
    - keypoints_image (numpy.ndarray): Image with the drawn keypoints and circles.
    """
    red = (0, 0, 255)
    blue = (255, 0, 0)
    keypoints_image = cv.cvtColor(image_iris.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(keypoints_image, centre, 2, red, thickness=3)
    cv.circle(keypoints_image, centre, iris_radius + 3, red, thickness=3)
    keypoints_image = cv.drawKeypoints(keypoints_image, keypoints, color=blue, flags=0, outImage=None)
    return keypoints_image

def angle(point_1, point_2):
    """
    Calculates the angle between two points relative to the x-axis.

    Parameters:
    - point_1 (tuple): Coordinates (x, y) of the first point.
    - point_2 (tuple): Coordinates (x, y) of the second point.

    Returns:
    - angle_360 (float): The angle in degrees between the two points, normalized to [0, 360).
    """
    x1, y1 = point_1
    x2, y2 = point_2
    angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    angle_360 = (angle + 360) % 360
    return angle_360

def to_polar(point, pole=(0, 0)):
    """
    Converts a Cartesian coordinate to polar coordinates.

    Parameters:
    - point (tuple): Cartesian coordinates (x, y).
    - pole (tuple): Origin of the polar coordinates (default is (0, 0)).

    Returns:
    - (r, theta): Polar coordinates (radius, angle).
    """
    r = distance(point, pole)
    theta = angle(point, pole)
    return r, theta

def normalize_r(r, pupil_radius, iris_radius):
    """
    Normalizes the radius value between the pupil and iris radii.

    Parameters:
    - r (float): The radius to normalize.
    - pupil_radius (float): The radius of the pupil.
    - iris_radius (float): The radius of the iris.

    Returns:
    - r_norm (float): The normalized radius between 0 and 1.
    """
    range = iris_radius - pupil_radius
    r_norm = (r - pupil_radius) / range
    return r_norm

def is_within_one_std(value, centre, dev):
    """
    Checks if a value lies within one standard deviation from the mean.

    Parameters:
    - value (float): The value to check.
    - centre (float): The mean or center value.
    - dev (float): The standard deviation.

    Returns:
    - bool: True if the value is within one standard deviation, False otherwise.
    """
    not_in_left_tail = value > centre - dev
    not_in_right_tail = value < centre + dev
    is_on_range = not_in_left_tail and not_in_right_tail
    return is_on_range

def warp_polar(image, x_size, y_size, centre, pupil_radius, iris_radius):
    """
    Warps the iris image into polar coordinates.

    Parameters:
    - image (numpy.ndarray): The iris image to warp.
    - x_size (int): The desired width of the output image.
    - y_size (int): The desired height of the output image.
    - centre (tuple): Coordinates (x, y) of the center of the iris.
    - pupil_radius (float): Radius of the pupil.
    - iris_radius (float): Radius of the iris.

    Returns:
    - normalized_iris (numpy.ndarray): The warped polar image.
    """

    normalized_iris=np.zeros(shape=(x_size, y_size))
    x_c, y_c = centre
        
    angle= 2.0 * math.pi / y_size
    inner_boundary_x = np.zeros(shape=(1, y_size))
    inner_boundary_y = np.zeros(shape=(1, y_size))
    outer_boundary_x = np.zeros(shape=(1, y_size))
    outer_boundary_y = np.zeros(shape=(1, y_size))

    for i in range(y_size):
        inner_boundary_x[0][i]= x_c + pupil_radius * math.cos(angle*(i))
        inner_boundary_y[0][i]= y_c + pupil_radius * math.sin(angle*(i))
        outer_boundary_x[0][i]= x_c + iris_radius * math.cos(angle*(i))
        outer_boundary_y[0][i]= y_c + iris_radius * math.sin(angle*(i))
        
    for j in range (y_size):
       for i in range (x_size):
            normalized_iris[i][j]= image[min(int(int(inner_boundary_y[0][j]) + (int(outer_boundary_y[0][j]) - int(inner_boundary_y[0][j])) * (i/64.0)),
                                        image.shape[0] - 1)][min(int(int(inner_boundary_x[0][j]) + (int(outer_boundary_x[0][j]) - int(inner_boundary_x[0][j])) * (i/64.0)),
                                        image.shape[1] - 1)]
            
    return normalized_iris

def gabor_filter(theta, config):
    """
    Creates a Gabor filter kernel for feature extraction.

    Parameters:
    - theta (float): The orientation of the filter.
    - config (object): Configuration object containing filter parameters.

    Returns:
    - kernel (numpy.ndarray): The generated Gabor filter kernel.
    """
    ksize = (config.gabor_filter.x_size, config.gabor_filter.y_size)
    kernel = cv.getGaborKernel(ksize, 
                                config.gabor_filter.gamma, 
                                theta,
                                config.gabor_filter.frequency,
                                config.gabor_filter.sigma,
                                config.gabor_filter.psi, 
                                ktype=cv.CV_64F)
    return kernel

def encoding(image, config):
    """
    Encodes an image using Gabor filters and extracts statistical features.

    Parameters:
    - image (numpy.ndarray): The iris image to encode.
    - config (object): Configuration object containing filter parameters.

    Returns:
    - vector (numpy.ndarray): The extracted feature vector.
    """
    vector = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
        gabor = gabor_filter(theta, config)
        filtered_eye = cv.filter2D(image, cv.CV_64F, gabor)
        for i in range(0, image.shape[0], 8):
            for j in range(0, image.shape[1], 8):                
                patch = filtered_eye[i:i+8, j:j+8]
                mean = patch.mean()
                AAD = np.abs(patch - mean).mean()
                vector.extend([mean, AAD]) 
    return np.array(vector)

def manage_best_model_and_metrics(model, evaluation_metric, metrics, best_metric, best_model, lower_is_better):
    """
    Updates the best model and metrics based on the evaluation metric.

    Parameters:
    - model (object): The model being evaluated.
    - evaluation_metric (str): The metric used for evaluation (e.g., 'accuracy').
    - metrics (dict): The metrics dictionary containing the model's performance.
    - best_metric (float): The current best metric value.
    - best_model (object): The current best model.
    - lower_is_better (bool): Whether a lower metric value is better (True for loss, False for accuracy).

    Returns:
    - best_model (object): The best performing model.
    - best_metric (float): The best performance metric.
    """
    if lower_is_better:
        is_best = metrics[evaluation_metric] < best_metric
    else:
        is_best = metrics[evaluation_metric] > best_metric
        
    if is_best:
        best_metric = metrics[evaluation_metric]
        best_model = model

    return best_model, best_metric

def accuracy_comparison1(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy):
    """
    Plots a bar chart comparing the accuracy of KNN, SVM, Neural Network models, and Merge Accuracy, and saves it to the 'graph' folder.

    :param accuracy_knn: Accuracy of the KNN model.
    :type accuracy_knn: float
    :param accuracy_svm: Accuracy of the SVM model.
    :type accuracy_svm: float
    :param accuracy_nn: Accuracy of the Neural Network model.
    :type accuracy_nn: float
    :param merge_accuracy: Combined accuracy of the models.
    :type merge_accuracy: float
    """
    models = ['KNN', 'SVM', 'Neural Network', 'Merge Accuracy']
    accuracies = [accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy]
    os.makedirs('graph', exist_ok=True)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'], alpha=0.8, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=12)

    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparison of Model Accuracies', fontsize=16)

    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join('graph', 'model_accuracy_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
def accuracy_comparison(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy, model_name="model"):
    """
    Plots a bar chart comparing the accuracy of KNN, SVM, Neural Network models, and Merge Accuracy,
    and saves it to the 'graph' folder with the model name included in the filename.
    """
    models = ['KNN', 'SVM', 'Neural Network', 'Merge Accuracy']
    accuracies = [accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy]
    os.makedirs('graph', exist_ok=True)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'], alpha=0.8, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=12)

    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Accuracy Comparison - {model_name}', fontsize=16)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f'model_accuracy_{model_name}.png'
    plot_path = os.path.join('graph', filename)
    plt.savefig(plot_path)
    plt.close()
    
def LLE_graph(X_train_red, y_train, save_path='hybrid_iris_recognition/graph/'):
    """
    Visualizes and saves the reduced features using LLE (Locally Linear Embedding).

    :param X_train_red: Reduced feature data.
    :param y_train: Labels associated with the data points.
    :param save_path: Directory path to save the plot (default is 'hybrid_iris_recognition/graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_red[:, 0], X_train_red[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
    
    plt.colorbar(scatter, label='Label')
    
    plt.title('Reduced Features Using LLE')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    
    plot_filename = os.path.join(save_path, 'reduced_features_lle.png')
    plt.savefig(plot_filename)
    plt.close() 
    
def identification_performance(tp, fp, tn, fn, save_path='hybrid_iris_recognition/graph/'):
    """
    Visualizes the performance of the identification system using a bar chart.

    :param tp: True Positives count.
    :param fp: False Positives count.
    :param tn: True Negatives count.
    :param fn: False Negatives count.
    :param save_path: Directory path to save the plot (default is 'hybrid_iris_recognition/graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    values = [tp, fp, tn, fn]
    
    plt.figure(figsize=(9, 6))
    plt.bar(categories, values, color=['green', 'red', 'blue', 'orange'])

    plt.title('Identification System Performance')
    plt.xlabel('Categories')
    plt.ylabel('Count')

    plot_filename = os.path.join(save_path, 'identification_performance.png')
    plt.savefig(plot_filename)
    plt.close()

def error_distribution_graph(frr, far, save_path='hybrid_iris_recognition/graph/'):
    """
    Visualizes the distribution of FRR and FAR using a pie chart.

    :param frr: False Rejection Rate.
    :param far: False Acceptance Rate.
    :param save_path: Directory path to save the plot (default is 'hybrid_iris_recognition/graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    labels = ['FRR', 'FAR']
    sizes = [frr, far]
    colors = ['red', 'blue']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Error Distribution (FRR vs FAR)')

    plot_filename = os.path.join(save_path, 'error_distribution.png')
    plt.savefig(plot_filename)
    plt.close()
   

def iris_code_plot(iris_code, path):
    """
    Plots the iris code and saves it as a JPEG image.

    Parameters:
    - iris_code: A list or array representing the iris code to be plotted.
    - path: The file path where the plot will be saved.
    """
    plt.plot(iris_code)
    plt.title(" IRIS CODE PLOT ")
    plt.xlabel("Index")
    plt.ylabel("Value")  
    plt.savefig(path, format="jpeg", dpi=300)
    plt.close()  
    

def plot_far_frr_vs_threshold(far, frr, thresholds, path):
    """
    Plots the FAR (False Acceptance Rate) and FRR (False Rejection Rate) curves
    against the thresholds and saves the plot as a JPEG image.

    Parameters:
    - far: A list or array representing the False Acceptance Rate values.
    - frr: A list or array representing the False Rejection Rate values.
    - thresholds: A list or array representing the thresholds used to calculate FAR and FRR.
    - path: The file path where the plot will be saved as a JPEG image.
    """
    plt.figure(figsize=(10, 5))  
    plt.plot(thresholds, far, marker='o', color='r', label="FAR")  
    plt.plot(thresholds, frr, marker='o', color='b', label="FRR")  
    plt.xlabel('Threshold')
    plt.ylabel('Percentage (%)')
    plt.title('FAR - FRR PLOT')  
    plt.grid(True)
    plt.legend(loc='best')  
    plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)
    plt.close()
