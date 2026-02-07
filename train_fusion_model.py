import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from collections import defaultdict
from Config import MODEL_DIR, BATCH_SIZE, RESULTS_DIR, MAX_EPOCHS, DROPOUT, SEQ_LEN
from results_utils import run_tests, plot_loss_curve, plot_accuracies
from load_data import DatasetManager
from datetime import datetime
from UnifiedFusionModel import UnifiedFusionModel
from shap_analysis import SHAP_UnifiedFusionModel
import random
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_model_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train_UnifiedFusionModel(seq_path, track_path, result_path, test_train_split_annotation_path, seq_features, track_features,
                             hidden_size=128, fusion_size=128, dropout=0.5, model_save_path="", test_prefix="no_prefix"):

    learning_rate = 1e-3
    print("[STEP 1] Loading and aligning data...")
    train_dataset = DatasetManager(seq_path, track_path, test_train_split_annotation_path, 0, seq_features, track_features)
    test_dataset = DatasetManager(seq_path, track_path, test_train_split_annotation_path, 1, seq_features, track_features, 
                                  train_dataset.seq_scaler_path, train_dataset.track_scaler_path, train_dataset.label_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=0, pin_memory=False)

    #Check if some labels are in the test sets but not in the training sets
    train_classes = set(torch.unique(train_dataset.labels).tolist())
    test_classes = set(torch.unique(test_dataset.labels).tolist())
    unknown_labels = test_classes - train_classes
    if (unknown_labels):
        raise Exception(f"Unknown labels label found in test set but not in train:", unknown_labels)

    classes, counts = np.unique(train_dataset.unencoded_labels, return_counts=True)
    print("Class counts (via numpy):")
    for cls, count in zip(classes, counts):
        print(f"  Class {cls}: {count} samples")

    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=train_dataset.unencoded_labels
    )

    # Convert to torch tensor
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class Weights:", weights)

    model = UnifiedFusionModel(seq_input_size=len(seq_features), track_input_size=len(track_features),
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-4)

    print("[STEP 2] Training unified fusion model...")
    best_acc, early_stop = 0, 0
    best_model = None

    train_loss_label, test_loss_label = "Train Loss", "Test Loss"
    train_accuracies_label, test_accuracies_label = "Train Accuracy", "Test Accuracy"

    losses_dict = {
        train_loss_label : [],
        test_loss_label : []
    }

    accuracies_dict = {
        train_accuracies_label : [],
        test_accuracies_label : []
    }

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(MAX_EPOCHS):
        model.train()
        correct_train_samples, total_train_samples, train_loss = 0, 0, 0
        correct_test_samples, total_test_samples, test_loss = 0, 0, 0
        for batch_seq, batch_track, batch_y, _prefix_tid in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()*batch_y.size(0)
            total_train_samples += batch_y.size(0)

            pred = logits.argmax(dim=1)
            correct_train_samples += (pred == batch_y).sum().item()

        model.eval()
        with torch.no_grad():
            for batch_seq, batch_track, batch_y, _prefix_tid in test_loader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)

                test_loss += loss.item() * batch_y.size(0)
                total_test_samples += batch_y.size(0)

                pred = logits.argmax(dim=1)
                correct_test_samples += (pred == batch_y).sum().item()

        train_loss /= total_train_samples
        train_acc = correct_train_samples / total_train_samples


        test_loss /= total_test_samples
        test_acc = correct_test_samples / total_test_samples

        scheduler.step(test_loss)
        
        if train_loss > 1.5:
            train_loss = 1.5
            if epoch > 10:
                break
        if test_loss >1.5:
            test_loss = 1.5
            if epoch > 10:
                break
        losses_dict[train_loss_label].append(train_loss)
        losses_dict[test_loss_label].append(test_loss)

        accuracies_dict[train_accuracies_label].append(train_acc)
        accuracies_dict[test_accuracies_label].append(test_acc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch + 1} | Loss = {train_loss:.4f} | Test Loss={test_loss:.4f}")
            print(f"Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}")
            print(scheduler.get_last_lr())

        if test_acc > best_acc:
            best_acc, best_model = test_acc, model.state_dict()
            early_stop = 0
        else:
            early_stop += 1

    # -------------------------------------------------
    # TRAINING LOSS/ACCURACY GRAPHS
    # -------------------------------------------------
    print("[STEP 3] Evaluating...")

    train_results_path = os.path.join(result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results")
    os.makedirs(train_results_path, exist_ok=True)

    model.load_state_dict(best_model)
    
    lstm_weight_norm = sum(p.norm().item() for n, p in model.lstm.named_parameters() if 'weight' in n)
    print("LSTM weight norm:", lstm_weight_norm)

    if model.track_input_size > 0:
        track_weight_norm = sum(p.norm().item() for n, p in model.track_fc.named_parameters() if 'weight' in n)
        print("Track weight norm:", track_weight_norm)

    np.savez(f"{train_results_path}/training_logs_unified.npz", 
             train_losses=losses_dict[train_loss_label], train_accs=accuracies_dict[train_accuracies_label],
             test_losses=losses_dict[test_loss_label], test_accuracies=accuracies_dict[test_accuracies_label])
    
    plot_loss_curve(losses_dict, train_results_path)
    plot_accuracies(accuracies_dict, train_results_path)
    
    test_results_path = os.path.join(result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/test results")
    os.makedirs(test_results_path, exist_ok=True)

    train_results = run_tests(model, train_dataset, train_loader, test_train_split_annotation_path, train_results_path, device)
    test_results = run_tests(model, test_dataset, test_loader, test_train_split_annotation_path, test_results_path, device)
    
    torch.save(model.state_dict(), os.path.join(train_results_path, f"hidden{hidden_size}_fusion{fusion_size}.pth" ))

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to", model_save_path)
    
    results_dict = {
        "train_losses": losses_dict[train_loss_label],
        "test_losses": losses_dict[test_loss_label],
        "train_accuracy": accuracies_dict[train_accuracies_label],
        "test_accuracy": accuracies_dict[test_accuracies_label],
    }

    results_dict = results_dict | train_results
    results_dict = results_dict | test_results

    return results_dict

def train_models_and_shap(ablation_configs, seq_dataset_path, track_dataset_path, test_train_split_annotation_path,
                 max_pow_hidden, max_pow_fusion, min_pow_hidden, min_pow_fusion, perform_SHAP_analysis = True):

    # === begin experiment ===
    for name, cfg in ablation_configs.items():
        print(f"\n===== Running Ablation: {name} =====")

        # model and dataset save route
        prefix = f"ablation_{name}"
        result_path = os.path.join(RESULTS_DIR, prefix)
        os.makedirs(result_path, exist_ok=True)
        
        train_accuracies_df = pd.DataFrame([["Train Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        test_accuracies_df = pd.DataFrame([["Test Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        train_test_loss_diff_df = pd.DataFrame([["Difference between Train/Val Loss"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
        r_squared_df = pd.DataFrame([["R-Squared (Test)"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])

        for fusion_pow in range(min_pow_fusion, max_pow_fusion):
            fusion_size = 2**fusion_pow
            train_accuracies = [fusion_size]
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
                    cfg["features"], cfg["track_features"],
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

                train_accuracies.append(metrics["train results"]["best_acc"])
                test_accuracies.append(metrics["test results"]["best_acc"])
                train_val_loss_diff.append((metrics["train_losses"][-1] - metrics["test_losses"][-1]))
                r_squareds.append(metrics["test results"]["r2"])
            
            train_accuracies_df.loc[len(train_accuracies_df)] = train_accuracies
            test_accuracies_df.loc[len(test_accuracies_df)] = test_accuracies
            train_test_loss_diff_df.loc[len(train_test_loss_diff_df)] = train_val_loss_diff
            r_squared_df.loc[len(r_squared_df)] = r_squareds

        empty_row = pd.DataFrame([[None] * train_accuracies_df.shape[1]], columns=train_accuracies_df.columns)
        all_accuracies_df = pd.concat([train_accuracies_df, empty_row,  
                                        train_test_loss_diff_df, empty_row,
                                        r_squared_df, empty_row,
                                        test_accuracies_df], ignore_index=True)


        now = datetime.now()
        month_day_time_str = now.strftime("%m_%d")
        hour_min_time_str = now.strftime("%H_%M")

        os.makedirs(os.path.join(RESULTS_DIR, now.strftime(f"{prefix}/accuracies/{month_day_time_str}")), exist_ok=True)
        all_accuracies_df.to_csv(os.path.join(result_path, f"accuracies/{month_day_time_str}/{hour_min_time_str}.csv"), index=False)