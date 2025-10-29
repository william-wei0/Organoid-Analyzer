import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from Config import *
from results_utils import *
from UnifiedFusionModel import UnifiedFusionModel
from shap_analysis import SHAP_UnifiedFusionModel
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


import random
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_model_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetDataset(Dataset):
    def __init__(self, seq_path, track_path, annotations_path, case_identifier, transform=None):
        X_seq, X_track, y_matched, prefix_tid = select_specific_cases(seq_path, track_path, annotations_path, case_identifier)

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_track = torch.tensor(X_track, dtype=torch.float32)
        self.prefix_tid = prefix_tid
        self.transform = transform  

    def __len__(self):
        # Total number of samples
        return len(self.prefix_tid)

    def __getitem__(self, idx):
        seq = self.X_seq[idx]
        track = self.X_track[idx]
        prefix_tid = self.prefix_tid[idx]

        # Apply optional transform to features
        if self.transform:
            seq, track = self.transform((seq, track))

        return seq, track, prefix_tid 

def select_specific_cases(seq_path, track_path, annotations_path, case_identifier):
    specfic_cases = []
    annotations_df = pd.read_excel(annotations_path)
    specfic_cases = annotations_df.loc[
        annotations_df["Train or Test"] == case_identifier, "Case"
    ].tolist()

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

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
            prefix_split = "_".join(tid[0].split("_")[:2])
            if prefix_split in specfic_cases:
                idx = track_id_to_index[key]
                X_seq_matched.append(X_seq[i])
                X_track_matched.append(X_track[idx])
                y_matched.append(y_seq[i])
                prefix_tid.append(tid[0]+str(tid[1]))

    return X_seq_matched, X_track_matched, y_matched, prefix_tid

def train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path):
    annotations_df = pd.read_excel(test_train_split_annotation_path)
    train_cases = annotations_df.loc[annotations_df["Train or Test"] == 0, "Case"].tolist()
    test_cases  = annotations_df.loc[annotations_df["Train or Test"] == 1, "Case"].tolist()

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("transposing...")
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }


    test_label_dist_dict = defaultdict(int)
    train_label_dist_dict = defaultdict(int)

    test_label_value_dist_dict = defaultdict(int)
    train_label_value_dist_dict = defaultdict(int)

    X_seq_matched_train, X_track_matched_train, y_matched_train, prefix_tid_train = [], [], [], []
    X_seq_matched_test, X_track_matched_test, y_matched_test, prefix_tid_test = [], [], [], []

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            case_name = "_".join(tid[0].split("_")[:2])

            if case_name in test_cases:
                X_seq_matched_test.append(X_seq[i])
                X_track_matched_test.append(X_track[idx])
                y_matched_test.append(y_seq[i])
                test_label_dist_dict[case_name] += 1
                test_label_value_dist_dict[y_seq[i]] += 1
            elif case_name in train_cases or not train_cases:
                X_seq_matched_train.append(X_seq[i])
                X_track_matched_train.append(X_track[idx])
                y_matched_train.append(y_seq[i])
                train_label_dist_dict[case_name] +=1
                train_label_value_dist_dict[y_seq[i]] += 1
            
    print(f"[DEBUG] Matched train pairs: {len(X_seq_matched_train)}")
    print(f"[DEBUG] Matched test pairs: {len(X_seq_matched_test)}")
    
    return (np.array(X_seq_matched_train), np.array(X_seq_matched_test), 
            np.array(X_track_matched_train), np.array(X_track_matched_test), 
            np.array(y_matched_train), np.array(y_matched_test))



def run_inference(model, X_seq, X_track, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_seq.to(device), X_track.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def normalize_dataset(X_seq, X_track, seq_scaler = None , track_scaler = None):
    # Helper functions
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

    # If no set scaler, create new seq_scaler and fit to training data
    if not seq_scaler:
        seq_scaler = StandardScaler()

        # Flatten sequence data (needed to use the scaler)
        n_samples, n_timesteps, n_features_seq = X_seq.shape
        X_seq_train_flat = X_seq.reshape(-1, n_features_seq)

        seq_scaler.fit(X_seq_train_flat)

    # If no set scaler, create new track_scaler and fit to training data
    if not track_scaler:
        track_scaler = StandardScaler()
        track_scaler.fit(X_track)

    X_seq_train_scaled = transform_seq(X_seq, seq_scaler)
    X_track_train_scaled = transform_track(X_track, track_scaler)

    return (X_seq_train_scaled, X_track_train_scaled, seq_scaler, track_scaler)

def Train_UnifiedFusionModel(seq_path, track_path, result_path, test_train_split_annotation_path,
                             seq_input_size=9, track_input_size=12, hidden_size=128, fusion_size=128, dropout=0.5, model_save_path="", test_prefix="no_prefix"):

    print("[STEP 1] Loading and aligning data...")
    X_seq_train_total, X_seq_test, X_track_train_total, X_track_test, y_train_total, y_test_original = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path=test_train_split_annotation_path)

    #Set labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_total)
    y_test = le.transform(y_test_original)

    #Check if some labels are in the test sets but not in the training sets
    train_classes = set(le.classes_)
    test_classes = set(y_test_original)
    unknown_labels = test_classes - train_classes
    if (unknown_labels):
        print("Unknown labels label found in test set but not in train:", unknown_labels)

    #Split train and validation cases
    (X_seq_train, X_seq_val, 
    X_track_train, X_track_val, 
    y_train, y_val) = train_test_split(X_seq_train_total, X_track_train_total, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Normalize Training Data and set Scaler
    X_seq_train, X_track_train, seq_scaler, track_scaler = normalize_dataset(X_seq_train, X_track_train)
    
    # Normalize Validation & Test Data using Scaler from Training Data
    X_seq_val, X_track_val, _seq_scal, _track_scal = normalize_dataset(X_seq_val, X_track_val, seq_scaler, track_scaler)
    X_seq_test, X_track_test, _seq_scal, _track_scal = normalize_dataset(X_seq_test, X_track_test, seq_scaler, track_scaler)

    #Transform to data to tensors
    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_val = torch.tensor(X_seq_val, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_val = torch.tensor(X_track_val, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create Tensor Datasets
    train_dataset = TensorDataset(X_seq_train, X_track_train, y_train_tensor)
    val_dataset = TensorDataset(X_seq_val, X_track_val, y_val_tensor)
    test_dataset = TensorDataset(X_seq_test, X_track_test, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)

    classes, counts = np.unique(y_train, return_counts=True)
    print("Class counts (via numpy):")
    for cls, count in zip(classes, counts):
        print(f"  Class {cls}: {count} samples")

    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_train
    )

    # Convert to torch tensor
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class Weights:", weights)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    print("[STEP 2] Training unified fusion model...")
    best_acc, early_stop = 0, 0
    entropy_total, entropy_count = 0.0, 0
    lowest_loss = 10000
    best_model = None
    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    test_losses, test_accs = [], []

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(MAX_EPOCHS):
        model.train()
        correct_train, total_train, train_loss = 0, 0, 0
        correct_val, total_val, val_loss = 0, 0, 0
        correct_test, total_test, test_loss = 0, 0, 0
        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct_train += (pred == batch_y).sum().item()
            total_train += batch_y.size(0)

        model.eval()
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in val_loader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct_val += (pred == batch_y).sum().item()
                total_val += batch_y.size(0)

            for batch_seq, batch_track, batch_y in test_loader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
                test_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct_test += (pred == batch_y).sum().item()
                total_test += batch_y.size(0)

        scheduler.step(val_loss)

        train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        test_loss /= len(test_loader)
        test_acc = correct_test / total_test
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch + 1} | Loss = {train_loss:.4f} | Val Loss={val_loss:.4f} | Test Loss={test_loss:.4f}")
            print(f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Test Acc={test_acc:.4f}")
            print(scheduler.get_last_lr())

        if val_loss < lowest_loss:
            lowest_loss, best_model = val_loss, model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 60:
                print("Early stopping triggered.")
                break

    # -------------------------------------------------
    # TRAINING LOSS/ACCURACY GRAPHS
    # -------------------------------------------------

    train_results_path = os.path.join(result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results")
    os.makedirs(train_results_path, exist_ok=True)

    model.load_state_dict(best_model)
    
    lstm_weight_norm = sum(p.norm().item() for n, p in model.lstm.named_parameters() if 'weight' in n)
    track_weight_norm = sum(p.norm().item() for n, p in model.track_fc.named_parameters() if 'weight' in n)
    print("LSTM weight norm:", lstm_weight_norm)
    print("Track weight norm:", track_weight_norm)
    
    
    np.savez(f"{train_results_path}/training_logs_unified.npz", 
             train_losses=train_losses, train_accs=train_accs,
             val_losses=val_losses, val_accs=val_accs,
             test_losses=test_losses, test_accuracies=test_accs)
    
    plot_loss_curve(train_losses, val_losses, test_losses, train_results_path)
    plot_accuracies(train_accs, val_accs, test_accs, train_results_path)
    

    print("[STEP 3] Evaluating...")

    # -------------------------------------------------
    # BEST TRAINING GRAPHS
    # -------------------------------------------------


    preds, probs = run_inference(model, X_seq_train, X_track_train, device)
    best_train_acc, f1, auc_value, y_train_bin = compute_metrics(y_train, preds, probs)

    print(f"[RESULT] Accuracy: {best_train_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")

    plot_roc(y_train_bin, probs, train_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_train, preds, le.classes_, train_results_path)

    fusion_weight_analysis(model, train_loader, device, train_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 0),
                                device, BATCH_SIZE, train_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, train_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")


    # -------------------------------------------------
    # BEST VALIDATION GRAPHS
    # -------------------------------------------------
    val_results_path = os.path.join(result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results/validation_")
    os.makedirs(val_results_path, exist_ok=True)

    preds, probs = run_inference(model, X_seq_val, X_track_val, device)
    best_val_acc, f1, auc_value, y_val_bin = compute_metrics(y_val, preds, probs)

    print(f"[RESULT] Accuracy: {best_val_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")

    plot_roc(y_val_bin, probs, val_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_val, preds, le.classes_, val_results_path)

    # -------------------------------------------------
    # BEST TEST GRAPHS
    # -------------------------------------------------
    test_results_path = os.path.join(result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/test results")
    os.makedirs(test_results_path, exist_ok=True)

    preds, probs = run_inference(model, X_seq_test, X_track_test, device)
    best_test_acc, f1, auc_value, y_test_bin = compute_metrics(y_test, preds, probs)

    print(f"[RESULT] Accuracy: {best_test_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")

    plot_roc(y_test_bin, probs, test_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_test, preds, le.classes_, test_results_path)

    fusion_weight_analysis(model, test_loader, device, test_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 1),
                                device, BATCH_SIZE, test_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, test_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")

    torch.save(model.state_dict(), os.path.join(train_results_path, f"hidden{hidden_size}_fusion{fusion_size}.pth" ))

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to", model_save_path)

    return {
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,

        "train_accuracy": train_accs,
        "val_accuracy": val_accs,
        "test_accuracy": test_accs,

        "best_train_acc": best_train_acc,
        "best_val_acc":best_val_acc,
        "best_test_acc": best_test_acc,
        
        "r2": r2
    }

def Test_UnifiedFusionModel(seq_path, track_path, model_path, test_train_split_annotation_path, results_dir="test", seq_input_size=9, track_input_size=12, hidden_size=128, fusion_size=128, dropout=0.5,):
    print("[TEST] Loading external test dataset...")

    
    X_seq_train_total, X_seq_test, X_track_train_total, X_track_test, y_train, y_test_original = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path=test_train_split_annotation_path)
    
    (X_seq_train, X_seq_val, 
    X_track_train, X_track_val, 
    y_train, y_val) = train_test_split(X_seq_train_total, X_track_train_total, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Normalize Training Data and set Scaler
    X_seq_train, X_track_train, seq_scaler, track_scaler = normalize_dataset(X_seq_train, X_track_train)
    
    # Normalize Test Data using Scaler from Training Data
    X_seq_test, X_track_test, _seq_scal, _track_scal = normalize_dataset(X_seq_test, X_track_test, seq_scaler, track_scaler)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test_original)

    train_classes = set(le.classes_)
    test_classes = set(y_test_original)
    unknown_labels = test_classes - train_classes
    if (unknown_labels):
        print("Unknown labels label found in test set but not in train:", unknown_labels)

    X_seq_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_track_tensor = torch.tensor(X_track_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_seq_tensor, X_track_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # -------------------------------------------------
    # Evaluation Graphs
    # -------------------------------------------------
    test_results_path = os.path.join(results_dir, f"test results/hidden_{hidden_size}/fusion_{fusion_size}")

    print("[STEP 3] Evaluating...")
    preds, probs = run_inference(model, X_seq_tensor, X_track_tensor, device)
    acc, f1, auc_value, y_test_bin = compute_metrics(y_test, preds, probs)

    print(f"[RESULT] Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    plot_roc(y_test_bin, probs, test_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_test, preds, le.classes_, test_results_path)

    fusion_weight_analysis(model, test_loader, device, test_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 1),
                                device, BATCH_SIZE, test_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, test_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")

    return {
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "acc": acc,
        "r2": r2
    }

def train_models_and_shap(ablation_configs, seq_dataset_path, track_dataset_path, test_train_split_annotation_path,
                 max_pow_hidden, max_pow_fusion, min_pow_hidden, min_pow_fusion, perform_SHAP_analysis = True):

    # === begin experiment ===
    for name, cfg in ablation_configs.items():
        print(f"\n===== Running Ablation: {name} =====")

        # model and dataset save route
        prefix = f"ablation_{name}"
        result_path = os.path.join(RESULTS_DIR, prefix)
        os.makedirs(result_path, exist_ok=True)

        seq_input_size = len(cfg["features"])
        track_input_size = len(cfg["track_features"])
        
        train_accuracies_df = pd.DataFrame([["Train Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        val_accuracies_df = pd.DataFrame([["Validation Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        test_accuracies_df = pd.DataFrame([["Test Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        train_val_loss_diff_df = pd.DataFrame([["Difference between Train/Val Loss"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        r_squared_df = pd.DataFrame([["R-Squared (Test)"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])

        for fusion_pow in range(min_pow_fusion, max_pow_fusion):
            fusion_size = 2**fusion_pow
            train_accuracies = [fusion_size]
            val_accuracies = [fusion_size]
            test_accuracies = [fusion_size]
            train_val_loss_diff = [fusion_size]
            r_squareds = [fusion_size]

            for hidden_pow in range(min_pow_hidden, max_pow_hidden):
                hidden_size = 2**hidden_pow
                print(f"Hidden Size: {hidden_size} | Fusion Size {fusion_size}")

                model_path = os.path.join(RESULTS_DIR, f"{prefix}/models/{prefix}_hidden{hidden_size}_fusion{fusion_size}.pth")
                os.makedirs(os.path.join(RESULTS_DIR, f"{prefix}/models"), exist_ok=True)

                metrics = Train_UnifiedFusionModel(
                    seq_dataset_path, track_dataset_path,
                    result_path,
                    test_train_split_annotation_path,
                    seq_input_size,track_input_size,
                    hidden_size, fusion_size,
                    DROPOUT, model_path, prefix
                )

                if perform_SHAP_analysis:
                    model_path = os.path.join(RESULTS_DIR, f"{prefix}/graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results/hidden{hidden_size}_fusion{fusion_size}.pth")
                    result_path = os.path.join(RESULTS_DIR, f"{prefix}/graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results")
                    SHAP_UnifiedFusionModel(
                        seq_length=SEQ_LEN,
                        features=cfg["features"],
                        track_features=cfg["track_features"],
                        model_save_path=model_path,
                        result_path=result_path,
                        seq_path=seq_dataset_path,
                        track_path=track_dataset_path,
                        hidden_size=hidden_size,
                        fusion_size=fusion_size
                    )

                train_accuracies.append(metrics["best_train_acc"])
                val_accuracies.append(metrics["best_val_acc"])
                test_accuracies.append(metrics["best_test_acc"])
                train_val_loss_diff.append(abs(metrics["train_losses"][-1] - metrics["val_losses"][-1])*-1)
                r_squareds.append(metrics["r2"])
            
            train_accuracies_df.loc[len(train_accuracies_df)] = train_accuracies
            val_accuracies_df.loc[len(val_accuracies_df)] = val_accuracies
            test_accuracies_df.loc[len(test_accuracies_df)] = test_accuracies
            train_val_loss_diff_df.loc[len(train_val_loss_diff_df)] = train_val_loss_diff
            r_squared_df.loc[len(r_squared_df)] = r_squareds

        empty_row = pd.DataFrame([[None] * train_accuracies_df.shape[1]], columns=train_accuracies_df.columns)
        all_accuracies_df = pd.concat([train_accuracies_df, empty_row, 
                                        val_accuracies_df, empty_row, 
                                        train_val_loss_diff_df, empty_row,
                                        r_squared_df, empty_row,
                                        test_accuracies_df], ignore_index=True)

        from datetime import datetime
        now = datetime.now()
        month_day_time_str = now.strftime("%m_%d")
        hour_min_time_str = now.strftime("%H_%M")

        os.makedirs(os.path.join(RESULTS_DIR, now.strftime(f"{prefix}/accuracies/{month_day_time_str}")), exist_ok=True)
        all_accuracies_df.to_csv(os.path.join(result_path, f"accuracies/{month_day_time_str}/{hour_min_time_str}.csv"), index=False)

