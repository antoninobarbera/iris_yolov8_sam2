import gc
import torch
from iris import iris_class
from identification import id_class
from tools.file_manager import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
from tqdm import tqdm

def generate_folder(config):
   """
   Generates the necessary folders for storing iris images and related data.
   """   
   if os.path.exists(config.folders_path.iris_images):
      shutil.rmtree(config.folders_path.iris_images)
   
   os.makedirs(config.folders_path.iris_images, exist_ok=True)
   os.makedirs(config.folders_path.original_iris, exist_ok=True)
   os.makedirs(config.folders_path.segmented_iris, exist_ok=True)
   os.makedirs(config.folders_path.keypoints_iris, exist_ok=True)
   os.makedirs(config.folders_path.mask_sam2, exist_ok=True)
   os.makedirs(config.folders_path.mask_iris, exist_ok=True)
   os.makedirs(config.folders_path.mask_pupil, exist_ok=True)
    

def load_irises(dataset, yolo_model, sam_predictor, config, device):
   """
   Processes a dataset of eye images to extract iris features using YOLOv8 for detection
   and SAM2 for segmentation, followed by feature extraction.

   Parameters:
   - dataset: A collection of eye images grouped by subjects/records.
   - yolo_model: The YOLOv8 model for eye/iris detection.
   - sam_predictor: The SAM2 predictor for iris segmentation.
   - config: Configuration object with model and dataset parameters.
   - device: Device to run the models on ('cuda' or 'cpu').

   Returns:
   - irises: A nested list where each record contains a list of iris_class objects for each subject.
   """
   irises = []

   for rec in tqdm(dataset, desc="Processing Records"):
      subjects = []
      for subject_idx, eye in enumerate(rec):
         iris_obtained = iris_class(eye, subject_idx, config, device)
         iris_obtained.segmentation(yolo_model, sam_predictor, config)
         iris_obtained.feature_extraction()
         subjects.append(iris_obtained)

      irises.append(subjects)

   return irises


def calculate_frr_far(irises, id, threshold):
   """
   Calculates the False Rejection Rate (FRR) and False Acceptance Rate (FAR).

   Parameters:
   - irises: A nested list of iris_class objects [record][subject].
   - id: An id_class object used for iris matching.
   - threshold: A similarity threshold for matching iris samples.

   Returns:
   - frr: The False Rejection Rate (percentage).
   - far: The False Acceptance Rate (percentage).
   """
   num_err_frr = 0
   num_err_far = 0
   tot_frr = 0
   tot_far = 0

   for subject_idx in tqdm(range(len(irises[0])), desc=f"FRR (Threshold={threshold})"):
      for i in range(len(irises)):
         for j in range(i + 1, len(irises)):
            match_x = id.sift_match(irises[i][subject_idx], irises[j][subject_idx], threshold)
            if not match_x:
               num_err_frr += 1
            tot_frr += 1
   for subj1 in tqdm(range(len(irises[0]) - 1), desc=f"FAR (Threshold={threshold})"):
      for subj2 in range(subj1 + 1, len(irises[0])):
         for rec1 in range(len(irises)):
            for rec2 in range(len(irises)):
               match_x = id.sift_match(irises[rec1][subj1], irises[rec2][subj2], threshold)
               if match_x:
                  num_err_far += 1
               tot_far += 1

   frr = num_err_frr / tot_frr * 100 if tot_frr > 0 else 0
   far = num_err_far / tot_far * 100 if tot_far > 0 else 0
   return round(frr, 2), round(far, 2)



if __name__ == '__main__':
   
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
   print('Start SIFT only system')
   print(f"\nLoading models:\n  - YOLOv8 model: {Path(checkpoint_yolov8n).name}\n  - SAM2 model: {Path(checkpoint_sam2).name}\n")
   
   
   if 'casiav1' in config.dataset_to_use:
      print("Loading CASIA v1 dataset...")
      dataset = load_dataset_casiav1(config)
      print('\n Original - Segmentation - Keypoints - Images Generation in progress...')
      irises = load_irises(dataset, yolo_model, sam_predictor, config, device)
         
   elif 'ubirisv1' in config.dataset_to_use:
      print("Loading Ubiris v1 dataset...")
      dataset = load_dataset_ubirisv1(config)
      print('\n Original - Segmentation - Keypoints - Images Generation in progress...')
      irises = load_irises(dataset, yolo_model, sam_predictor, config, device)
         
   elif 'casia_v3_interval' in config.dataset_to_use:
      print("Loading Casia-v3-Interval dataset...")
      dataset = load_dataset_casia_interval(config)
      print('\n Original - Segmentation - Keypoints - Images Generation in progress...')
      irises = load_irises(dataset, yolo_model, sam_predictor, config, device)
      
   elif 'casia_v3_lamp' in config.dataset_to_use:
      print("Loading Casia-v3-Lamp dataset...")
      create_casia_lamp_pickle()
      dataset = load_dataset_casia_lamp(config)
      print('\n Original - Segmentation - Keypoints - Images Generation in progress...')
      irises = load_irises(dataset, yolo_model, sam_predictor, config, device)
      
   else:
      print('Dataset not recognized. Please check the configuration file.')
      sys.exit(1)
      
   id = id_class(config, None)
   thresholds = config.test_thresholds
   
   print('-----------------------------------------------------------------')
   for threshold in thresholds:
      print("\nTesting Threshold " + str(threshold) + ':')
      frr, far = calculate_frr_far(irises, id, threshold)
      print(" FAR : " + str(far) + " %")
      print(" FRR : " + str(frr) + " %")
      print('-----------------------------------------------------------------\n')
      
      
      
      
      
   '''# --- CALCOLO TEMPO IDENTIFICAZIONE PER OGNI UTENTE ---
   print("\nCalcolo tempo identificazione per ogni utente...")
   threshold = thresholds[0]  # Usa la prima soglia come esempio
   user_times = {}

   for user_idx in range(len(irises[0])):  # ciclo su ogni soggetto
      start_time = time.time()
      for rec_idx in range(len(irises)):
            for compare_rec in range(len(irises)):
               if rec_idx != compare_rec:
                  id.sift_match(irises[rec_idx][user_idx], irises[compare_rec][user_idx], threshold)
      end_time = time.time()
      elapsed_time = end_time - start_time
      user_times[user_idx] = elapsed_time
      print(f"Soggetto {user_idx}: tempo identificazione = {elapsed_time:.5f} secondi")
   
   # Media del tempo di identificazione
   avg_time = sum(user_times.values()) / len(user_times)
   print(f"\nTempo medio di identificazione per utente: {avg_time:.5f} secondi")'''
   