import os
import cv2
import numpy as np
from tools.file_manager import *


def center_between_left_edges(iris_bbox, pupil_bbox):
    """
    Calculate the center point between the left edges of the iris and pupil bounding boxes.

    Args:
        iris_bbox (list): Bounding box coordinates for the iris in the format [x1, y1, x2, y2].
        pupil_bbox (list): Bounding box coordinates for the pupil in the format [x1, y1, x2, y2].

    Returns:
        tuple: (x_center, y_center) - The center coordinates between the left edges of the iris and pupil bounding boxes.
    """
    x1_iris, y1_iris, x2_iris, y2_iris = iris_bbox
    x1_pupil, _, _, _ = pupil_bbox

    x_center = (x1_iris + x1_pupil) / 2
    y_center = (y1_iris + y2_iris) / 2

    return x_center, y_center
    
    
def infer_yolo_bbox(model, image, conf_thres=0.10):
    """
    Run inference using YOLO to detect iris and pupil in the image.

    Args:
        model: The YOLO model for object detection.
        image: The input image in BGR format.
        conf_thres: Confidence threshold for detections.

    Returns:
        - best_iris: Bounding box coordinates for the detected iris.
        - best_pupil: Bounding box coordinates for the detected pupil.
        - iris_point: Coordinates of the iris point.
        - pupil_point: Coordinates of the pupil point.
        - pupil_circle_center: Center of the pupil circle.
        - pupil_circle_radius: Radius of the pupil circle.
        - iris_circle_radius: Radius of the iris circle.
    """
    img = image

    results = model(img, conf=conf_thres, verbose=False)
    detections = results[0].boxes

    bboxes = detections.xyxy.cpu().numpy()
    classes = detections.cls.cpu().numpy()      
    scores = detections.conf.cpu().numpy()      
    names = model.names                         

    best_iris = None
    best_pupil = None
    best_iris_conf = -1
    best_pupil_conf = -1
    y_offset = 10

    for box, cls, conf in zip(bboxes, classes, scores):
        class_name = names[int(cls)]
        coords = [float(coord) for coord in box]

        if class_name == 'iris' and conf > best_iris_conf:
            best_iris = coords
            best_iris_conf = conf

        elif class_name == 'pupil' and conf > best_pupil_conf:
            best_pupil = coords
            best_pupil_conf = conf
            
    if best_iris and best_pupil:
        x_c, y_c = center_between_left_edges(best_iris, best_pupil)
        x_c += 10
        y_c += 20
        iris_point = (x_c, y_c)
    elif best_iris:
        x_c += 10
        y_c += 20
        iris_point = (x_c, y_c)
    else:
        iris_point = None

    if best_pupil:
        x1_p, y1_p, x2_p, y2_p = best_pupil
        x_c_p = int((x1_p + x2_p) / 2)
        y_c_p = int((y1_p + y2_p) / 2) + y_offset
        pupil_point = (x_c_p, y_c_p)
    else:
        pupil_point = None

    # Addition: inscribed circles
    iris_circle_center = None
    iris_circle_radius = None
    pupil_circle_center = None
    pupil_circle_radius = None

    if best_iris:
        x1_i, y1_i, x2_i, y2_i = best_iris
        cx_i = (x1_i + x2_i) / 2
        cy_i = (y1_i + y2_i) / 2
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        r_i = min(w_i, h_i) / 2
        iris_circle_center = (int(cx_i), int(cy_i))
        iris_circle_radius = int(r_i)

    if best_pupil:
        x1_p, y1_p, x2_p, y2_p = best_pupil
        cx_p = (x1_p + x2_p) / 2
        cy_p = (y1_p + y2_p) / 2
        w_p = x2_p - x1_p
        h_p = y2_p - y1_p
        r_p = min(w_p, h_p) / 2
        pupil_circle_center = (int(cx_p), int(cy_p))
        pupil_circle_radius = int(r_p)
  
    return best_iris, best_pupil, iris_point, pupil_point
    
    
def infer_yolo_bbox_test(model, image, conf_thres=0.10):
    img = image.copy()

    save_vis=True
    
    results = model(img, conf=conf_thres, verbose=False)
    detections = results[0].boxes

    bboxes = detections.xyxy.cpu().numpy()
    classes = detections.cls.cpu().numpy()
    scores = detections.conf.cpu().numpy()
    names = model.names

    best_iris = None
    best_pupil = None
    best_iris_conf = -1
    best_pupil_conf = -1
    y_offset = 10

    for box, cls, conf in zip(bboxes, classes, scores):
        class_name = names[int(cls)]
        coords = [float(coord) for coord in box]

        if class_name == 'iris' and conf > best_iris_conf:
            best_iris = coords
            best_iris_conf = conf

        elif class_name == 'pupil' and conf > best_pupil_conf:
            best_pupil = coords
            best_pupil_conf = conf

    iris_point = None
    pupil_point = None

    if best_iris and best_pupil:
        x_c, y_c = center_between_left_edges(best_iris, best_pupil)
        x_c += 10
        y_c += 20
        iris_point = (int(x_c), int(y_c))
    elif best_iris:
        x1_iris, y1_iris, x2_iris, y2_iris = best_iris
        x_c = (x1_iris + x2_iris) / 2 + 10
        y_c = (y1_iris + y2_iris) / 2 + 20
        iris_point = (int(x_c), int(y_c))

    if best_pupil:
        x1_p, y1_p, x2_p, y2_p = best_pupil
        x_c_p = int((x1_p + x2_p) / 2)
        y_c_p = int((y1_p + y2_p) / 2) + y_offset
        pupil_point = (x_c_p, y_c_p)

    if save_vis:
        annotated_img = image.copy()

        # Bounding box
        if best_iris:
            x1, y1, x2, y2 = map(int, best_iris)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, "Iris", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if best_pupil:
            x1, y1, x2, y2 = map(int, best_pupil)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, "Pupil", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Points
        if iris_point:
            cv2.circle(annotated_img, iris_point, 4, (0, 255, 255), -1) 
            cv2.putText(annotated_img, "Iris Pt", (iris_point[0] + 5, iris_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if pupil_point:
            cv2.circle(annotated_img, pupil_point, 4, (0, 0, 255), -1)
            cv2.putText(annotated_img, "Pupil Pt", (pupil_point[0] + 5, pupil_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        save_dir = os.path.join("iris_images", "yolo_annotated")
        os.makedirs(save_dir, exist_ok=True)
        
        existing_files = os.listdir(save_dir)
        existing_ids = [
            int(f.split(".")[0]) for f in existing_files
            if f.endswith(".jpg") and f.split(".")[0].isdigit()
        ]
        next_id = max(existing_ids) + 1 if existing_ids else 0
        save_path = os.path.join(save_dir, f"{next_id}.jpg")
        cv2.imwrite(save_path, annotated_img)

    iris_circle_center = None
    iris_circle_radius = None
    pupil_circle_center = None
    pupil_circle_radius = None

    if best_iris:
        x1_i, y1_i, x2_i, y2_i = best_iris
        cx_i = (x1_i + x2_i) / 2
        cy_i = (y1_i + y2_i) / 2
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        r_i = min(w_i, h_i) / 2
        iris_circle_center = (int(cx_i), int(cy_i))
        iris_circle_radius = int(r_i)

    if best_pupil:
        x1_p, y1_p, x2_p, y2_p = best_pupil
        cx_p = (x1_p + x2_p) / 2
        cy_p = (y1_p + y2_p) / 2
        w_p = x2_p - x1_p
        h_p = y2_p - y1_p
        r_p = min(w_p, h_p) / 2
        pupil_circle_center = (int(cx_p), int(cy_p))
        pupil_circle_radius = int(r_p)

    return best_iris, best_pupil, iris_point, pupil_point, pupil_circle_center, pupil_circle_radius, iris_circle_radius


def infer_sam2(sam_predictor, image_rgb, bbox_iris, bbox_pupil, iris_point, pupil_point, config):
    """
    Run inference using SAM2 to segment the iris and pupil from the image.

    Args:
        - sam_predictor: The SAM2 predictor object.
        - image_rgb: The input image in RGB format.
        - bbox_iris: Bounding box for the iris.
        - bbox_pupil: Bounding box for the pupil.
        - iris_point: Coordinates of the iris point.
        - pupil_point: Coordinates of the pupil point.
        - config: Configuration object containing parameters for processing.

    Returns:
        - addweighted_image: The final processed image with enhanced contrast.
        - mask_iris_filled: The filled mask for the iris.
        - mask_pupil_filled: The filled mask for the pupil.
    """
    mask_iris_filled = None
    mask_pupil_filled = None
    addweighted_image = None

    if bbox_iris:
        input_prompt = np.array(bbox_iris)
        point_coords = np.array([iris_point])
        point_labels = np.array([1])
        masks_iris, _, _ = sam_predictor.predict(box=input_prompt, point_coords=point_coords, point_labels=point_labels, multimask_output=False)
        mask_iris = (masks_iris[0] * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_iris_filled = cv2.morphologyEx(mask_iris, cv2.MORPH_CLOSE, kernel)

    if bbox_pupil:
        input_prompt = np.array(bbox_pupil)
        point_coords = np.array([pupil_point])
        point_labels = np.array([1])
        masks_pupil, _, _ = sam_predictor.predict(box=input_prompt, point_coords=point_coords, point_labels=point_labels, multimask_output=False)
        mask_pupil = (masks_pupil[0] * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_pupil_filled = cv2.morphologyEx(mask_pupil, cv2.MORPH_CLOSE, kernel)

    if mask_iris_filled is not None:
        masked_iris = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_iris_filled)

        if mask_pupil_filled is not None:
            mask_pupil_inv = cv2.bitwise_not(mask_pupil_filled)
            iris_without_pupil = cv2.bitwise_and(masked_iris, masked_iris, mask=mask_pupil_inv)
        else:
            iris_without_pupil = masked_iris
        
        final_image = cv2.cvtColor(iris_without_pupil, cv2.COLOR_RGB2GRAY)
        #final_image = add_light(final_image, gamma=1.8)
        equalized_image = cv2.equalizeHist(final_image)
        addweighted_image = cv2.addWeighted(final_image, config.equalization.alpha, equalized_image, config.equalization.beta, config.equalization.gamma)
        
    return addweighted_image, mask_iris_filled, mask_pupil_filled


def add_light(image, gamma):
    """
    Add light to the image using gamma correction.
    """
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    return cv2.LUT(image, table)


def detect_circle_contours(mask):
    """
    Detect the largest contour in the mask and return its center and radius.

    Args:
        mask (numpy.ndarray): The binary mask image where contours are to be detected.

    Returns:
        tuple: Center (x, y) and radius of the largest contour, or None if no contour is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        return (int(x), int(y)), int(radius)
    return None, None


def draw_circle_on_mask(mask, center, radius, color=(255, 255, 255)):
    """
    Draw a circle on the mask at the specified center and radius.

    Args:
        mask (numpy.ndarray): The binary mask image where the circle will be drawn.
        center (tuple): Center of the circle (x, y).
        radius (int): Radius of the circle.
        color (tuple): Color of the circle in BGR format (default is white).

    Returns:
        numpy.ndarray: The mask with the circle drawn on it.
    """
    if mask is None or center is None or radius is None:
        return mask

    if len(mask.shape) == 2:
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_color = mask.copy()

    cv2.circle(mask_color, center, radius, color, 2)
    cv2.circle(mask_color, center, 3, (0, 0, 255), -1)
    return mask_color
    

def crop_pad_resize_and_adjust_coords(image, bbox, pupil_center, pupil_radius, iris_radius, target_size=256):
    """
    Crop, pad, and resize the image based on the bounding box and adjust coordinates of the pupil and iris.

    Args:
        image (numpy.ndarray): The input image to be processed.
        bbox (list): Bounding box coordinates in the format [x1, y1, x2, y2].
        pupil_center (tuple): Center of the pupil (x, y).
        pupil_radius (int): Radius of the pupil.
        iris_radius (int): Radius of the iris.
        target_size (int): The target size for resizing the image.

    Returns:
        tuple: Resized image, adjusted pupil center, adjusted pupil radius, adjusted iris radius.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]

    h, w = cropped.shape[:2]
    max_side = max(h, w)

    delta_w = max_side - w
    delta_h = max_side - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    scale_x = target_size / max_side
    scale_y = target_size / max_side

    if pupil_center is not None:
        cx_old, cy_old = pupil_center
        cx_new = (cx_old - x1 + left) * scale_x
        cy_new = (cy_old - y1 + top) * scale_y
        new_pupil_center = (int(cx_new), int(cy_new))
    else:
        new_pupil_center = None

    new_pupil_radius = int(pupil_radius * scale_x) if pupil_radius else None
    new_iris_radius = int(iris_radius * scale_x) if iris_radius else None

    return resized, new_pupil_center, new_pupil_radius, new_iris_radius


#### SEGMENTATION INFERENCE FUNCTION ####
def yolo_sam_inference(image, yolo_model, sam_predictor, config, idx):
    """
    Run inference using YOLO for bounding box detection and SAM2 for segmentation.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        yolo_model: The YOLO model for bounding box detection.
        sam_predictor: The SAM2 predictor object for segmentation.
        config: Configuration object containing parameters for processing.
        idx: Index or identifier for the image.
        
    Returns:
        - seg_img: The segmented image after processing.
        - pupil_center: Center coordinates of the pupil.
        - pupil_radius: Radius of the pupil.
        - iris_radius: Radius of the iris.
    """
    #YOLO inference
    bbox_iris, bbox_pupil, iris_point, pupil_point = infer_yolo_bbox(yolo_model, image)

    # Check if bounding boxes are detected
    if bbox_iris is None or bbox_pupil is None:
        print(f"Immagine {idx}: IRIDE o PUPILLA NON RILEVATA")

        # Save the image in a separate folder
        path = config.folders_path.not_detected
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(path, image)
        return None, None, None, None
    
    #SAM2 inference
    image_bgr = image.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)
    
    seg_img, mask_iris, mask_pupil = infer_sam2(
                                            sam_predictor=sam_predictor,
                                            image_rgb=image_rgb,
                                            bbox_iris=bbox_iris,
                                            bbox_pupil=bbox_pupil,
                                            iris_point=iris_point,
                                            pupil_point=pupil_point,
                                            config=config
                                            )
    
    iris_center, iris_radius = detect_circle_contours(mask_iris)
    pupil_center, pupil_radius = detect_circle_contours(mask_pupil)
        
    seg_img, pupil_center, pupil_radius, iris_radius = crop_pad_resize_and_adjust_coords(
                                                            seg_img,
                                                            bbox=bbox_iris,
                                                            pupil_center=pupil_center,
                                                            pupil_radius=pupil_radius,
                                                            iris_radius=iris_radius,
                                                            target_size=256
                                                            )
    
    return seg_img, pupil_center, pupil_radius, iris_radius, mask_iris, mask_pupil
