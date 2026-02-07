import numpy as np
import pandas as pd
import torch
import joblib
import os
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from Config import GENERATED_DIR, SEQ_FEATURES, TRACK_FEATURES
from sklearn.preprocessing import LabelEncoder
    
class DatasetManager(Dataset):
    def __init__(self, seq_path, track_path, annotations_path, case_identifier, 
                 selected_seq_features = SEQ_FEATURES, selected_track_features = TRACK_FEATURES, 
                 seq_scaler_path = None, track_scaler_path = None, label_encoder = None, transform=None):
        

        X_seq, X_track, labels_matched, prefix_tid = self.select_specific_cases(seq_path, track_path, annotations_path, case_identifier, selected_seq_features, selected_track_features)
        X_seq, X_track, seq_scaler_path, track_scaler_path = self.normalize_dataset(X_seq, X_track, seq_scaler_path, track_scaler_path)
        encoded_labels, label_encoder = self.encode_labels(labels_matched, label_encoder)

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_track = torch.tensor(X_track, dtype=torch.float32)
        self.labels = torch.tensor(encoded_labels, dtype=torch.long)
        self.unencoded_labels = labels_matched

        self.prefix_tid = prefix_tid
        self.transform = transform
        self.seq_scaler_path = seq_scaler_path
        self.track_scaler_path = track_scaler_path
        self.label_encoder = label_encoder
        self.case_identifier = case_identifier
        self.selected_seq_features = selected_seq_features
        self.selected_track_features = selected_track_features

    def __len__(self):
        return len(self.prefix_tid)

    def __getitem__(self, idx):
        seq = self.X_seq[idx]
        track = self.X_track[idx]
        label = self.labels[idx]
        prefix_tid = self.prefix_tid[idx]

        # Apply optional transform to features
        if self.transform:
            seq, track = self.transform((seq, track))

        return seq, track, label, prefix_tid
    
    def encode_labels(self, labels, label_encoder = None):
        if label_encoder:
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
        return encoded_labels, label_encoder
    
    def select_specific_cases(self, seq_path, track_path, annotations_path, case_identifier, 
                              selected_seq_features = SEQ_FEATURES, selected_track_features = TRACK_FEATURES):
        def select_features_from_dataset(feature_dataset, features, selected_features):
            feature_to_idx = {seq_feature: index for index, seq_feature in enumerate(features)}
            cols = [feature_to_idx[feature] for feature in selected_features]
            if feature_dataset.ndim == 3:
                return feature_dataset[:, :, cols]
            else:
                return feature_dataset[:, cols]
            
        annotations_df = pd.read_excel(annotations_path)
        specfic_cases = annotations_df.loc[annotations_df["Train or Test"] == case_identifier, "Case"].tolist()

        seq_data = np.load(seq_path, allow_pickle=True)
        track_data = np.load(track_path, allow_pickle=True)

        X_seq, y_seq, track_ids_seq, seq_feature_list = seq_data['X'], seq_data['y'], seq_data['track_ids'], list(seq_data['feature_list'])
        X_track, y_track, track_ids_track, track_feature_list = track_data['X'], track_data['y'], track_data['track_ids'], list(track_data['feature_list'])
        X_seq = select_features_from_dataset(X_seq, seq_feature_list, selected_seq_features)
        X_track = select_features_from_dataset(X_track, track_feature_list, selected_track_features)

        if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
            print("transposing...")
            X_seq = np.transpose(X_seq, (0, 2, 1))

        track_id_to_index = {
            tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
            for i, tid in enumerate(track_ids_track)
        }

        X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []
        for i, tid in enumerate(track_ids_seq):
            key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
            if key in track_id_to_index:
                case_name = "_".join(tid[0].split("_")[:2])
                
                if case_name in specfic_cases:
                    # if case_identifier == 0 and case_name == "PDO_Device8":
                    #     print(y_seq[i], end = " ")
                    idx = track_id_to_index[key]
                    X_seq_matched.append(X_seq[i])
                    X_track_matched.append(X_track[idx])
                    y_matched.append(y_seq[i])
                    prefix_tid.append(tid[0]+str(tid[1]))
        return np.array(X_seq_matched), np.array(X_track_matched), np.array(y_matched), prefix_tid
    
    def normalize_dataset(self, X_seq, X_track, seqential_scaler_path = None , track_scaler_path = None):
        def transform_seq(X_seq, scaler):
            if X_seq is None:
                return None
            n_samples, n_timesteps, n_features_seq = X_seq.shape
            X_flat = X_seq.reshape(-1, n_features_seq)
            X_scaled = scaler.transform(X_flat)
            return X_scaled.reshape(n_samples, n_timesteps, n_features_seq)

        def transform_track(X_track, scaler):
            if X_track is None:
                return None
            return scaler.transform(X_track)

        scaler_save_folder = f"{GENERATED_DIR}/normalization_scalers"
        # If no set scaler, create new seq_scaler, fit to training data, and save 
        if not seqential_scaler_path:
            seqential_scaler = RobustScaler()

            # Flatten sequence data (needed to use the scaler)
            n_samples, n_timesteps, n_features_seq = X_seq.shape
            X_seq_train_flat = X_seq.reshape(-1, n_features_seq)
            seqential_scaler.fit(X_seq_train_flat)

            #Save scaler
            seqential_scaler_path = f"{scaler_save_folder}/seqential_data_scaler.joblib"
            os.makedirs(scaler_save_folder, exist_ok=True)
            joblib.dump(seqential_scaler, seqential_scaler_path)


        # If no set scaler, create new track_scaler, fit to training data, and save 
        if not track_scaler_path:
            track_scaler = RobustScaler()
            track_scaler.fit(X_track)

            track_scaler_path = f"{scaler_save_folder}/track_data_scaler.joblib"
            os.makedirs(scaler_save_folder, exist_ok=True)
            joblib.dump(track_scaler, track_scaler_path)
        
        seqential_scaler = joblib.load(seqential_scaler_path)
        track_scaler = joblib.load(track_scaler_path)
        
        X_seq_train_scaled = transform_seq(X_seq, seqential_scaler)
        X_track_train_scaled = transform_track(X_track, track_scaler)

        return (X_seq_train_scaled, X_track_train_scaled, seqential_scaler_path, track_scaler_path)








