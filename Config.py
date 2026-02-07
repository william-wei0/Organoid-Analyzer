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

DATASET_CONFIGS = {
    "CARTLOW1": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\CARTLOW1_images",
                "annotation_path" : f"{DATA_DIR}/CARTLOW1_annotation.xlsx",
                "data_folder" : f"{DATA_DIR}/CARTLOW1",
                "case_name": "CARTLOW1",
                "prefix": "",
                "subcase_names" : ["NCI9Stroma7",
                                   "NCI9Stroma8"],
                "specific_image_transformations" : {
                    "Despeckle" : ["DiD-CAR T_NCI9Stroma7\XY1"],
                    "Smooth" : ["DiD-CAR T_NCI9Stroma7\XY1"],
                    "specific_thresholds" : {"DiD-CAR T_NCI9Stroma8\XY5": 120}
                },
            },
    "CARTLOW2": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\CARTLOW2_images", 
                "annotation_path" : f"{DATA_DIR}/CARTLOW2_annotation.xlsx",
                "data_folder" : f"{DATA_DIR}/CARTLOW2",
                "case_name": "CARTLOW2",
                "prefix": "",
                "subcase_names" : ["NCI9Stroma7",
                                    "NCI9Stroma8"]},
    "CART": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\CART_images", 
             "annotation_path" : f"{DATA_DIR}/CART annotations.xlsx", 
             "data_folder" : f"{DATA_DIR}/CART",
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
            "specific_image_transformations" : {
                "specific_thresholds": {"DiD-MSLN_NCI6_5 percent 20ms001\XY4" : 130}
            },
        },


    "2nd": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\2ND_images",
            "annotation_path" : f"{DATA_DIR}/2nd batch annotations.xlsx",
                "data_folder" : f"{DATA_DIR}/2ND",
                 "case_name": "2ND",
                    "prefix": "2nd_",
            "subcase_names" : ["NYU352",
                               "NYU360",
                               "NCI6",
                               "NCI8",
                               "NCI9"]},
    
    "PDO": {"images_folder" : r"C:\Users\billy\Documents\VIP Images\PDO_images",
            "annotation_path" : f"{DATA_DIR}/PDO_annotation.xlsx",
                "data_folder" : f"{DATA_DIR}/PDO",
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

SEQ_DATASET_PREFIX = ""
TRACK_DATASET_PREFIX = ""

SEQ_FEATURES = [ # Time-based Features 
    'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 
    'SPEED', "MEAN_SQUARE_DISPLACEMENT", 
    'RADIUS', "ELLIPSE_THETA", "ELLIPSE_MINOR", "ELLIPSE_MAJOR",
    "POSITION_X", "POSITION_Y", "SHAPE_INDEX",
]


TRACK_FEATURES = [ # Track-Level Statistics Features
    "TRACK_DISPLACEMENT", "TRACK_STD_SPEED",
    "MEAN_DIRECTIONAL_CHANGE_RATE", "TOTAL_DISTANCE_TRAVELED", 
    "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO"
]

FEATURE_LEN = len(SEQ_FEATURES)
TRACK_LEN = len(TRACK_FEATURES)


# ======= TRAINING SETTINGS =======
TEST_TRAIN_SPLIT_ANNOTATION_PATH = r"./Data/Annotations.xlsx"
SEQ_DATASET_PATH = os.path.join(GENERATED_DIR, f"{SEQ_DATASET_PREFIX}trajectory_dataset_{SEQ_LEN}.npz")
TRACK_DATASET_PATH = os.path.join(GENERATED_DIR, f"{TRACK_DATASET_PREFIX}track_dataset.npz")

DROPOUT = 0.0
MAX_EPOCHS = 400
BATCH_SIZE = 256
MIN_POW_FUSION = 2
MAX_POW_FUSION = 3

MIN_POW_HIDDEN = 3
MAX_POW_HIDDEN = 8

ABLATION_CONFIGS = {
    "Specify" : {
        "features": SEQ_FEATURES,
        "track_features" :TRACK_FEATURES
    },
}

