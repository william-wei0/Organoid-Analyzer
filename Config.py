import os

# ======= GENERATE FOLDERS =======

# Upload this folder with original data files
DATA_DIR = "./Data"
os.makedirs(DATA_DIR, exist_ok=True)
# Manually delete this folder before uploading
GENERATED_DIR = "./Generated"
os.makedirs(GENERATED_DIR, exist_ok=True)
MODEL_DIR = os.path.join(GENERATED_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
# Upload this folder with results and plots
RESULTS_DIR = "./Results"


# ======= CELL TRACKING SETTINGS =======
FIJI_PATH = r"C:\Users\billy\Desktop\Fiji.app"
JAVA_ARGUMENTS = '-Xmx12g'

SPECIAL_THRESHOLDING = {"DiD-MSLN_NCI6_5 percent 20ms001\XY4" : 130}
CELL_TRACKING_DATASET_CONFIGS = {
    "CART": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\William_20250429_8 different patient_day 6 CART_for AI", 
                  "case_name": "CART",
                     "prefix": "",
             "subcase_names" : ["NYU318",
                                "NYU352",
                                "NYU358",
                                "NYU360",
                                "NCI2",
                                "NCI6",
                                "NCI8",
                                "NCI9"],
        "specific_thresholds": {"DiD-MSLN_NCI6_5 percent 20ms001\XY4" : 130}},

    "2nd": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\William_20250522_Meso IL18 CART_5 patients_day 6_for AI", 
                 "case_name": "2ND",
                    "prefix": "2nd_",
            "subcase_names" : ["NYU352",
                               "NYU360",
                               "NCI6",
                               "NCI8",
                               "NCI9"]},
    
    "PDO": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\William_20250710_PDO device 1 to 8_for AI", 
                 "case_name": "PDO",
                    "prefix": "",
            "subcase_names" : ["Device1",
                               "Device2",
                               "Device3",
                               "Device4",
                               "Device5",
                               "Device6",
                               "Device7",
                               "Device8"]},
}





# ======= DATASET GENERATION SETTINGS =======
SEQ_LEN = 100 # Number of frames to use.
DATASET_CONFIGS = {
    "CART": {"annotation_path" : f"{DATA_DIR}/CART annotations.xlsx", 
                 "data_folder" : f"{DATA_DIR}/CART"},

    "2nd": {"annotation_path" : f"{DATA_DIR}/2nd batch annotations.xlsx",
                "data_folder" : f"{DATA_DIR}/2ND"},
    
    "PDO": {"annotation_path" : f"{DATA_DIR}/PDO_annotation.xlsx",
                "data_folder" : f"{DATA_DIR}/PDO"},
}

SEQ_DATASET_PREFIX = ""
TRACK_DATASET_PREFIX = ""

features = [ # Time-based Features 
    'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_ASPECTRATIO','SOLIDITY', 
    'SPEED', "MEAN_SQUARE_DISPLACEMENT", #"RADIUS"
]


track_features = [ # Track-Level Statistics Features
    "TRACK_DISPLACEMENT", "TRACK_STD_SPEED",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

FEATURE_LEN = len(features)
TRACK_LEN = len(track_features)


# ======= TRAINING SETTINGS =======
TEST_TRAIN_SPLIT_ANNOTATION_PATH = r"C:\Users\billy\Desktop\VIP\Organoid-Analyzer\Data\Annotations.xlsx"
SEQ_DATASET_PATH = os.path.join(GENERATED_DIR, f"{SEQ_DATASET_PREFIX}trajectory_dataset_{SEQ_LEN}.npz")
TRACK_DATASET_PATH = os.path.join(GENERATED_DIR, f"{TRACK_DATASET_PREFIX}track_dataset.npz")

DROPOUT = 0.3
MAX_EPOCHS = 400
BATCH_SIZE = 256

MIN_POW_FUSION = 4
MAX_POW_FUSION = 12

MIN_POW_HIDDEN = 2
MAX_POW_HIDDEN = 7

ABLATION_CONFIGS = {
    "Specify" : {
        "features": features,
        "track_features" :track_features
    },
}

