# Organoid Analyzer

A Python-based machine learning pipeline for analyzing time-lapse microscopy images of organoids using cell tracking and deep learning classification.

## Overview

This project uses Fiji's TrackMate plugin to track cells in time-lapse images, extracts features from the trajectories, and trains a unified fusion model (combining LSTM sequence features and track-level features) to classify organoid responses into three categories: Progressive, Stable, or Responsive.

---

## Table of Contents

- [Installation](#installation)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Scripts](#scripts)
- [Model Architecture](#model-architecture)
- [Output Files](#output-files)
- [Requirements](#requirements)
- [Support](#support)

---

## Installation

### 1. Download the Repository

Download the project from GitHub:
- Go to [https://github.com/william-wei0/Organoid-Analyzer](https://github.com/william-wei0/Organoid-Analyzer)
- Click **Code > Download ZIP**
- Extract the ZIP file to a new folder

### 2. Install Python 3.10.11

Download and install Python 3.10.11 from:
[https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)

### 3. Set Up Virtual Environment

1. **Open the project in VSCode**:
   - File > Open Folder (select the folder with Python files)

2. **Open Command Prompt terminal**:
   - Terminal > New Terminal
   - Click dropdown arrow next to "powershell" > Select **Command Prompt**

3. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Install PyTorch**:
   - Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - **If you have an NVIDIA GPU**: Select CUDA 12.6 (or latest)
   - **If you don't have an NVIDIA GPU**: Select CPU
   - Copy and run the installation command in the terminal
   - Example (CUDA): 
     ```bash
     pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     ```

### 4. Running the Code

To run any script, activate the virtual environment and use:
```bash
python filename.py
```

**Note**: The virtual environment must be active each time you run the code.

---

## Workflow

### Expected Pipeline:

1. **Configure settings** in `Config.py`
2. **Track cells** using `track_cells.py`
3. **Define annotation loading** in `create_dataset.py` (`load_annotations()` function)
4. **Generate dataset** with `create_dataset.py`
5. **Set up train/test split** in Excel annotation sheet
6. **Train and test models** with `train_and_test_models.py`

---

## Configuration

All configuration settings are centralized in `Config.py`.

### Cell Tracking Settings

| Parameter | Description |
|-----------|-------------|
| `IMAGES_FOLDER` | Path to folder containing time-lapse images |
| `FIJI_PATH` | Path to Fiji folder (not the app itself, just the directory) |
| `CASE_NAME` | Name of the output folder for storing raw tracking results in the DATA folder |
| `JAVA_ARGUMENTS` | Java memory allocation settings for Fiji (increase for large images) |

### Dataset Generation Settings

| Parameter | Description |
|-----------|-------------|
| `SEQ_LEN` | Number of frames to use per sequence |
| `DATASET_CONFIGS` | Dictionary mapping case names to annotation paths and data folders |
| `SEQ_DATASET_PREFIX` | Name prefix for output sequential dataset files |
| `TRACK_DATASET_PREFIX` | Name prefix for output track dataset files |
| `features` / `track_features` | Lists of features to compute and include in the dataset |

**Example `DATASET_CONFIGS`**:
```python
DATASET_CONFIGS = {
    "CART": {
        "annotation_path": f"{DATA_DIR}/CART annotations.xlsx",
        "data_folder": f"{DATA_DIR}/CART"
    }
}
```

### Training Settings

| Parameter | Description |
|-----------|-------------|
| `TEST_TRAIN_SPLIT_ANNOTATION_PATH` | Excel sheet defining which cases are used for training vs testing |
| `SEQ_DATASET_PATH` / `TRACK_DATASET_PATH` | Paths to specific datasets (optional override) |
| `DROPOUT` | Dropout rate for regularization (e.g., 0.3 = 30%) |
| `MAX_EPOCHS` | Maximum number of training epochs (early stopping may end training sooner) |
| `BATCH_SIZE` | Training batch size (larger = faster but more memory) |
| `MIN_POW_FUSION` / `MAX_POW_FUSION` | Range of fusion layer sizes (powers of 2, e.g., 2-10 = 4, 8, 16...1024) |
| `MIN_POW_HIDDEN` / `MAX_POW_HIDDEN` | Range of LSTM hidden sizes (powers of 2) |
| `ABLATION_CONFIGS` | Dictionary for testing different feature combinations without creating new datasets |

---

## Scripts

### `track_cells.py`

Processes time-lapse image folders using Fiji's TrackMate plugin to automatically detect and track cells.

**Key Steps**:
1. Check and create output folder
2. Launch Fiji with TrackMate
3. Load images and convert to hyperstacks
4. Configure detection and tracking parameters (with automatic thresholding)
5. Run TrackMate and extract spot/track features
6. Save results as CSV files

**Outputs**:
- `tracks.csv`: Track-level features for each detected trajectory
- `spots.csv`: Frame-by-frame spot features

---

### `create_dataset.py`

Generates training datasets from TrackMate outputs.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `load_annotations` | Reads annotation Excel files and maps sample IDs to labels (0, 0.5, 1.0) |
| `load_tracks_and_spots` | Loads and merges track/spot CSVs with labels |
| `filter_valid_trajectories` | Removes trajectories shorter than minimum length |
| `compute_features` | Calculates velocity, speed, direction, MSD from positions |
| `compute_msd` | Computes mean squared displacement for different time lags |
| `align_and_save_dataset` | Pads/trims sequences to fixed length, saves as .npz and .csv |
| `build_track_level_dataset` | Creates track-level feature dataset |
| `filter_outer` | Removes invalid trajectories based on shape features |
| `save_unscaled_spot_features` | Saves unscaled spot features to CSV |
| `save_unscaled_track_features` | Saves unscaled track features to CSV |

**Outputs**:
- `.npz` files: Compressed datasets for model training
- `.csv` files: Human-readable datasets for inspection

---

### `train_fusion_model.py`

Trains the unified fusion model combining sequence and track features.

**Key Components**:

| Component | Description |
|-----------|-------------|
| `SubsetDataset` | PyTorch dataset class for loading case-specific data |
| `select_specific_cases` | Filters and aligns sequence/track data by case annotations |
| `train_test_split_by_case` | Splits data into train/test sets based on case annotations |
| `run_inference` | Runs model in evaluation mode and returns predictions |
| `Train_UnifiedFusionModel` | Main training function with early stopping, model saving, and evaluation |
| `Test_UnifiedFusionModel` | Evaluates pre-trained models on test data |
| `train_models_and_shap` | Runs ablation studies across multiple configurations |

**Features**:
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)
- Comprehensive evaluation (accuracy, F1, AUC, confusion matrices)
- Case-level analysis
- Optional SHAP analysis for interpretability

---

### `shap_analysis.py`

Performs SHAP (SHapley Additive exPlanations) analysis for model interpretability.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `load_and_align_data` | Loads and aligns sequence/track data by track IDs |
| `SHAP_UnifiedFusionModel` | Computes and visualizes feature importance using SHAP |

**Outputs**:
- CSV files with SHAP values
- Bar charts of feature importance (signed and absolute)
- Time-summed feature importance plots
- SHAP summary plots

---

### `results_utils.py`

Utility functions for generating evaluation metrics and visualizations.

**Key Functions**:

| Function | Description |
|----------|-------------|
| `compute_metrics` | Calculates accuracy, F1-score, and ROC AUC |
| `plot_roc` | Plots ROC curves for each class |
| `plot_confusion_matrix` | Creates confusion matrix heatmap |
| `fusion_weight_analysis` | Tests model performance across different sequence/track weight balances |
| `compute_case_proportions` | Visualizes predicted class distributions per case |
| `correlate_with_size_change` | Correlates predictions with experimental size change data |
| `plot_loss_curve` | Plots training and validation loss over epochs |
| `plot_accuracies` | Plots training, validation, and test accuracies over epochs |

---

### `UnifiedFusionModel.py`

Defines the neural network architecture.

**Model Components**:

1. **Attention Module**
   - Learns to focus on important time steps in sequences
   - Uses softmax-normalized attention weights
   - Applies dropout for regularization
   - Computes context vector as weighted sum of LSTM outputs

2. **Sequence Encoder (BiLSTM + Attention + LayerNorm)**
   - Bidirectional LSTM captures past and future context
   - Layer normalization stabilizes training
   - Attention reduces variable-length sequences to fixed-size vectors
   - Final layer normalization before fusion

3. **Track Encoder (Optional)**
   - Fully connected network: Linear → ReLU → Dropout → Linear
   - Maps track features to same dimension as LSTM output
   - Skipped if no track features are provided

4. **Fusion Layer**
   - Combines sequence and track features with learnable weights
   - Scales features using `lstm_weight` vs `1 - lstm_weight`
   - Concatenates weighted representations

5. **Classifier Head**
   - Fully connected block: Linear → ReLU → Dropout → Linear
   - Output dimension: 3 (Progressive, Stable, Responsive)

---

## Model Architecture

The Unified Fusion Model combines two types of features:

### Sequence Features (Time-series)
- Extracted via BiLSTM + Attention from cell trajectories over time
- Captures temporal dynamics and movement patterns
- Features include: position, velocity, speed, direction, MSD

### Track Features (Static)
- Aggregate statistics from entire tracks
- Features include: duration, mean speed, total displacement, track length
- Provides global context about trajectory behavior

Both feature types are weighted, fused, and passed through a classifier for 3-class prediction.

---

## Output Files

### Tracking Outputs
- **`tracks.csv`**: Track-level features (duration, speed, displacement, etc.)
- **`spots.csv`**: Frame-by-frame spot features (position, intensity, shape, etc.)

### Dataset Files
- **`.npz` files**: Compressed datasets for training (sequence and track features)
- **`.csv` files**: Human-readable datasets for inspection

### Training Outputs
- **Model checkpoints**: Saved best models during training (`.pt` files)
- **Performance metrics**: Accuracy, F1, AUC scores
- **Confusion matrices**: Classification error analysis
- **ROC curves**: Class separation visualization
- **Loss curves**: Training and validation loss over epochs
- **Accuracy curves**: Training, validation, and test accuracy over epochs

### Analysis Outputs
- **SHAP plots**: Feature importance visualizations
- **Case-level analysis**: Predicted class distributions per case
- **Correlation plots**: Predictions vs experimental size change data
- **Ablation study results**: Performance across different configurations

---

## Requirements

- **Python**: 3.10.11
- **PyTorch**: CUDA or CPU version (see installation)
- **Fiji**: For TrackMate cell tracking
- **Additional packages**: See `requirements.txt` for full list

### Key Python Packages
- PyTorch (for deep learning)
- NumPy (for numerical operations)
- Pandas (for data manipulation)
- Scikit-learn (for evaluation metrics)
- SHAP (for model interpretability)
- Matplotlib/Seaborn (for visualizations)
- JPype (for Fiji integration)

---

## Support

For issues, questions, or contributions, please visit the GitHub repository:
[https://github.com/william-wei0/Organoid-Analyzer](https://github.com/william-wei0/Organoid-Analyzer)

---

## License

[GPL-3.0]

---

## Acknowledgments

This project uses:
- **Fiji/ImageJ** for image processing
- **TrackMate** for cell tracking
- **PyTorch** for deep learning
- **SHAP** for model interpretability