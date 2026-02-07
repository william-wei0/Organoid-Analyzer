from train_fusion_model import train_models_and_shap
from create_dataset import create_dataset
from Config import *

def ablation_tests_features():
    seq_features = []
    seq_features_names = []
    ablation_config = {}

    for seq_index, seq_feature in enumerate(SEQ_FEATURES):
        seq_features.append(seq_feature)
        seq_features_names.append(seq_feature[:3])

        track_features = []
        track_features_names = []

        for track_index, track_feature in enumerate(TRACK_FEATURES):
            track_features.append(track_feature)
            track_features_names.append(track_feature[:3])
            test_name = ", ".join(seq_features_names + track_features_names)
            ablation_config[test_name] = {"features": seq_features.copy(), "track_features" :track_features.copy()}

    train_models_and_shap(ablation_config, SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        MAX_POW_HIDDEN, MAX_POW_FUSION, MIN_POW_HIDDEN, MIN_POW_FUSION, perform_SHAP_analysis = False)
    

def ablation_tests_features_remove_one():
    ablation_config = {}

    for seq_index, seq_feature in enumerate(SEQ_FEATURES):
        seq_features = SEQ_FEATURES.copy()
        seq_features.remove(seq_feature)
        ablation_config[seq_feature] = {"features": seq_features.copy(), "track_features" :TRACK_FEATURES}
    for track_index, track_feature in enumerate(TRACK_FEATURES):
        track_features = TRACK_FEATURES.copy()
        track_features.remove(track_feature)
        ablation_config[track_feature] = {"features": SEQ_FEATURES, "track_features" :track_features.copy()}
    train_models_and_shap(ablation_config, SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        MAX_POW_HIDDEN, MAX_POW_FUSION, MIN_POW_HIDDEN, MIN_POW_FUSION, perform_SHAP_analysis = False)


if __name__ == "__main__":
    ablation_tests_features_remove_one()
