import os
import sys
import cv2
import yaml
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from addict import Dict
from pathlib import Path
from yaml_config_override import add_arguments


def directory_exists(path):
    """
       directory_exists() checks if a directory truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The directory path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the directory exists.
    """
    if os.path.isdir(path):
        return True  
    else:
        return False
        
        
def file_exists(file_path, message):
    """
       file_exists() checks if a file truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The file path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the file exists.
    """
    if os.path.isfile(file_path):
        return True
    else: 
        print(message)
        sys.exit()


def move_directory(source, destination):
    """
       move_directory() moves a directory into another directory.

         Args:
         source: The path of the directory that will be moved.
         destination: The destination directory path.
    """
    path = os.path.join(destination, source)
    if os.path.isdir(path): 
        shutil.rmtree(path)
    shutil.move(source, destination)

    
def configuration(main_path=None):
    """
       configuration() extract from the configuration file (config\base_config.yaml) the configuration parameters
       and save it in a Dict object. Also it checks if the configuration file exists.

        Args:
        main_path(path string): The path of the parent folder. Default value: None. 

        Returns:
        config (addict Dict): Dict containing configuration parameters.
    """
    # Set path.
    relative_path = os.path.join('config', 'base_config.yaml')
    if main_path is None: 
        path = relative_path
    else:
        path = os.path.join(main_path, relative_path)

    file_exists(path, " NO CONFIGURATION FILE DETECTED --> ADD A base_config.yaml FILE TO config DIRECTORY ")
    config_source = yaml.safe_load(Path(path).read_text())
    config = Dict(add_arguments(config_source))
    return config


#### CASIA V1 dataset ####
def load_dataset_casiav1(config):
    """
    Loads the CASIA dataset from the 'dataset' directory.
    
    Args:
    - config (addict.Dict): Configuration object containing parameters.
    
    Returns:
    - list: A list of records from the CASIA dataset.
    """
    casia_path = config.dataset.casiav1
        
    if not os.path.isfile(casia_path):
        print(f"Error: The file '{casia_path}' does not exist.")
        sys.exit()
        
    casia_path = config.dataset.casiav1
        
    try:
        with open(casia_path, 'rb') as file:
            casia = pickle.load(file)
    except Exception as e:
        print(f"ERROR trying to load '{casia_path}'.")
        print(f"Exception details: {str(e)}")
        sys.exit()

    rec_1 = casia['rec_1']
    rec_2 = casia['rec_2']
    rec_3 = casia['rec_3']
    rec_4 = casia['rec_4']
    rec_5 = casia['rec_5']
    rec_6 = casia['rec_6']
    rec_7 = casia['rec_7']

    return [rec_1, rec_2, rec_3, rec_4, rec_5, rec_6, rec_7]


#### UBIRIS V1 dataset ####
def load_dataset_ubirisv1(config):
    """
    Loads the UBIRIS dataset from the 'dataset' directory.
    
    Returns:
    - list: a list of N lists (records), each containing images of different subjects.
    """
    ubiris_path = config.dataset.ubirisv1

    if not os.path.isfile(ubiris_path):
        print(f"Error: The file '{ubiris_path}' does not exist.")
        sys.exit()

    try:
        with open(ubiris_path, 'rb') as file:
            dataset = pickle.load(file)
    except Exception as e:
        print(f"ERROR trying to load '{ubiris_path}'.")
        print(f"Exception details: {str(e)}")
        sys.exit()

    # Se è un dict, trasformalo in lista ordinata come negli altri dataset
    if isinstance(dataset, dict):
        dataset = [dataset[k] for k in sorted(dataset.keys())]

    return dataset


#### CASIA V3 INTERVAL dataset ####
def create_casia_interval_pickle():
    """
    Creates a pickle file for the CASIA Iris Interval dataset.
    """
    input_root = Path("dataset/casia_interval")
    output_path = Path("dataset/CASIA_V3_INTERVAL.pkl")

    dataset = {i: [] for i in range(5)}

    print("Creating pickle CASIA Iris Interval...")

    user_folders = sorted(
        [f for f in input_root.iterdir() if f.is_dir()],
        key=lambda x: int(x.name)
    )

    for user_folder in tqdm(user_folders, desc="Cartelle utenti"):
        image_paths = sorted(
            [p for p in user_folder.glob("*") if p.suffix.lower() in [".jpg", ".png", ".bmp", ".jpeg"]],
            key=lambda p: p.name
        )

        if len(image_paths) != 5:
            print(f"[ATTENTION] {user_folder.name} it only has {len(image_paths)} images (expect at least 5)")
            continue

        for idx, img_path in enumerate(image_paths):
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[ERROR] Unable to read image: {img_path}")
                continue
            dataset[idx].append(img)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nPickle salvato in: {output_path.resolve()}")

def load_dataset_casia_interval(config):
    """
    Loads the CASIA Iris Interval dataset from a pickle file.
    
    Args:
    - config (addict.Dict): Configuration object containing parameters.
    
    Returns:
    - list: A list of 5 lists, each containing the images from the corresponding position.
    """
    interval_path = config.dataset.casia_v3_interval

    if not os.path.isfile(interval_path):
        print(f"Error: The file '{interval_path}' does not exist.")
        sys.exit()

    try:
        with open(interval_path, 'rb') as file:
            dataset = pickle.load(file)
    except Exception as e:
        print(f"ERROR trying to load '{interval_path}'.")
        print(f"Exception details: {str(e)}")
        sys.exit()

    return [dataset[i] for i in range(5)]


#### CASIA V3 LAMP dataset ####
def create_casia_lamp_pickle():
    """
    Creates a pickle file for the CASIA Iris Lamp dataset.
    """
    input_root = Path("dataset/casia_lamp")
    output_path = Path("dataset/CASIA_V3_LAMP.pkl")

    dataset = {i: [] for i in range(20)}

    print("Creating pickle dataset...")

    user_folders = sorted(
        [f for f in input_root.iterdir() if f.is_dir()],
        key=lambda x: int(x.name)
    )

    for user_folder in tqdm(user_folders, desc="User Folders"):
        image_paths = sorted(
            [p for p in user_folder.glob("*") if p.suffix.lower() in [".jpg", ".png", ".bmp", ".jpeg"]],
            key=lambda p: p.name
        )

        if len(image_paths) < 20:
            print(f"[ATTENTION] {user_folder.name} it only has {len(image_paths)} images (expect at least 20)")
            continue

        for idx in range(20):
            img_path = image_paths[idx]
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[ERROR] Unable to read image: {img_path}")
                continue
            dataset[idx].append(img)
            
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nPickle saved at: {output_path.resolve()}")
    
def load_dataset_casia_lamp(config):
    """
    Loads the CASIA-Iris-Lamp dataset from a pickle file.

    Args:
    - config (addict.Dict): Configuration object containing dataset path.

    Returns:
    - list: A list of 20 lists, each containing images from one position (0-19).
    """
    lamp_path = config.dataset.casia_v3_lamp

    if not os.path.isfile(lamp_path):
        print(f"Error: The file '{lamp_path}' does not exist.")
        sys.exit()

    try:
        with open(lamp_path, 'rb') as file:
            dataset = pickle.load(file)
    except Exception as e:
        print(f"ERROR trying to load '{lamp_path}'.")
        print(f"Exception details: {str(e)}")
        sys.exit()

    return [dataset[i] for i in range(20)]


#### UBIRIS V2 (session 2) dataset ####
def create_sessao2():
    """
    Creates a pickle file for the UBIRIS V2 (session 2) dataset.
    """
    input_root = Path("dataset/sessao2")
    output_path = Path("dataset/sessao2.pkl")

    dataset = {0: []}

    print("Creating pickle dataset...")

    image_paths = sorted(
        [p for p in input_root.glob("*") if p.suffix.lower() in [".tif", ".tiff"]],
        key=lambda p: p.name
    )

    if not image_paths:
        print("[ERROR] No images found in the directory.")
        return

    for img_path in tqdm(image_paths, desc="Lettura immagini"):
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"[ERROR] Unable to read image: {img_path}")
            continue

        dataset[0].append(img)

    # Salva il dataset in un file pickle
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nPickle saved at: {output_path.resolve()}")

def load_dataset_sessao2(config):
    """
    Loads the UBIRIS V2 (session 2) dataset from a pickle file.

    Args:
        config (addict.Dict): Configuration object containing dataset path.

    Returns:
        list: List containing all images (dataset[0]).
    """
    sessao2_path = config.dataset.sessao2_path

    if not isinstance(sessao2_path, (str, os.PathLike)):
        print(f"[ERROR] Invalid path: {sessao2_path}")
        sys.exit()

    if not os.path.isfile(sessao2_path):
        print(f"[ERROR] File '{sessao2_path}' non esiste.")
        sys.exit()

    try:
        with open(sessao2_path, 'rb') as file:
            dataset = pickle.load(file)
    except Exception as e:
        print(f"[ERROR] Unable to load pickle file '{sessao2_path}'.")
        print(f"Details: {str(e)}")
        sys.exit()

    if 0 not in dataset:
        print(f"[ERROR] Key '0' not found in the dataset.")
        sys.exit()

    return dataset[0]


#### CASIA V2 TWINS dataset ####
def create_casia3_twins():
    """
    Creates a pickle file for the CASIA V3 Twins dataset.
    """
    input_root = Path("dataset/casia3_twins")
    output_path = Path("dataset/CASIA_V3_TWINS.pkl")

    dataset = {0: []}

    print("Creating pickle for CASIA V3 Twins dataset...")

    image_paths = sorted(
        [p for p in input_root.rglob("*") if p.suffix.lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]],
        key=lambda p: p.name
    )

    if not image_paths:
        print("[ERROR] No images found in the directory.")
        return

    for img_path in tqdm(image_paths, desc="Reading images"):
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"[ERROR] Unable to read image: {img_path}")
            continue

        dataset[0].append(img)

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nPickle saved at: {output_path.resolve()}")

def load_dataset_casia3_twins(config):
    """
    Loads the CASIA V3 Twins dataset from a pickle file.

    Args:
        config (addict.Dict): Configuration object containing the dataset path.

    Returns:
        list: List containing all images (dataset[0]).
    """
    casia3_twins_path = config.dataset.casia3_twins_path

    if not isinstance(casia3_twins_path, (str, os.PathLike)):
        print(f"[ERROR] Invalid path: {casia3_twins_path}")
        sys.exit()

    if not os.path.isfile(casia3_twins_path):
        print(f"[ERROR] File '{casia3_twins_path}' does not exist.")
        sys.exit()

    try:
        with open(casia3_twins_path, 'rb') as file:
            dataset = pickle.load(file)
    except Exception as e:
        print(f"[ERROR] Unable to load pickle file '{casia3_twins_path}'.")
        print(f"Details: {str(e)}")
        sys.exit()

    if 0 not in dataset:
        print(f"[ERROR] Key '0' not found in the dataset.")
        sys.exit()

    return dataset[0]


def load_casiav1_flat(config):
    """
    Loads CASIA v1 returning all images in a single flat list.
    """
    records = load_dataset_casiav1(config)
    flat_list = [img for record in records for img in record if img is not None]
    return flat_list


def load_ubirisv1_flat(config):
    """
    Loads UBIRIS V1 returning all images in a single flat list.
    """
    records = load_dataset_ubirisv1(config)
    flat_list = [img for record in records for img in record if img is not None]
    return flat_list


def load_casia_interval_flat(config):
    """
    Loads CASIA Interval returning all images in a single flat list.
    """
    records = load_dataset_casia_interval(config) 
    flat_list = [img for record in records for img in record if img is not None]
    return flat_list


def load_casia_lamp_flat(config):
    """
    Loads CASIA Lamp returning all images in a single flat list.
    """
    records = load_dataset_casia_lamp(config)
    flat_list = [img for record in records for img in record if img is not None]
    return flat_list
