import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Config import GENERATED_DIR, features, track_features, SEQ_LEN, DATASET_CONFIGS, SEQ_DATASET_PREFIX, TRACK_DATASET_PREFIX

class Dataset_Batch:
    def __init__(self, annotation_path, data_folder, mapping = {}):
        self.annotation_path = annotation_path
        self.data_folder = data_folder
        self.mapping = mapping


def save_unscaled_spot_features(spots_df, output_prefix=""):
    # extract unscaled features and save to CSV
    unscaled_df = spots_df[["PREFIX", "TRACK_ID", "FRAME", "LABEL"] + features].copy()

    out_path = os.path.join(GENERATED_DIR, f"unscaled_spot_features{output_prefix}.csv")
    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled spot features to: {out_path}")

def save_unscaled_track_features(tracks_df, datasets, output_prefix=""):
    # extract unscaled track features and save to CSV
    merged_mapping = {}

    for label in datasets:
        overlap = merged_mapping.keys() & datasets[label].mapping.keys()  # intersection of keys
        if overlap:
            raise ValueError(f"Overlapping keys found when merging: {overlap}")
        merged_mapping.update(datasets[label].mapping)

    def match_label(prefix, merged_mapping):
        prefix = "_".join(prefix.split("_")[:2])
        return merged_mapping.get(prefix, np.nan)

    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label, args=(merged_mapping,))
    unscaled_df = tracks_df[["PREFIX", "TRACK_ID", "LABEL"] + track_features].copy()

    out_path = os.path.join(GENERATED_DIR, f"unscaled_track_features{output_prefix}.csv")

    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled track features to: {out_path}")


# === Step 1: Load Annotations ===
def load_annotations(path, folder):

    folder = folder.split("/")[-1]
    match folder:
        case "CART":
            df = pd.read_excel(path, sheet_name="Summary")
            id_series = df.iloc[:, 1].astype(str).str.strip()
            label_series = df.iloc[:, 2].astype(float)
        case "2ND":
            df = pd.read_excel(path, sheet_name=0)
            id_series = df["Meso IL18 CAR T cells"].astype(str).str.strip()
            label_series = df["Labels"].astype(float)
        case "PDO":
            df = pd.read_excel(path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "")
            id_series = df["Name"].astype(str).str.replace(" ", "")
            label_series = df["Score"]
        case "CAF":
            df = pd.read_excel(path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "")
            id_series = df["Name"].astype(str).str.replace(" ", "")
            label_series = df["Score"]
        case "DiffStroma":
            df = pd.read_excel(path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "")
            id_series = df["Name"].astype(str).str.replace(" ", "")
            label_series = df["Score"]

        case _:
            raise Exception(f"Undefined Data loading type: {folder}. Specify how to load the annotations in load_annoations.")
            
    mapping = dict(zip(id_series, label_series))
    mapping = {f"{folder}_{device}": label for device, label in mapping.items()}
    print(f"Loaded annotation from {path}")
    print("Total entries:", len(mapping))
    print(mapping)
    return mapping


# === Step 2: Load Track/Spot Files ===
def load_tracks_and_spots(datasets):
    spots = []
    tracks = []

    for label in datasets:

        for file_name in os.listdir(datasets[label].data_folder):
            if not file_name.endswith("_tracks.csv"):
                continue
            
            prefix = file_name.replace("_tracks.csv", "")
            spot_file = file_name.replace("_tracks.csv", "_spots.csv")

            track_path = os.path.join(datasets[label].data_folder, file_name)
            spot_path = os.path.join(datasets[label].data_folder, spot_file)

            label_dict = datasets[label].mapping
            prefix_split = prefix.split("_")

            folder_name = datasets[label].data_folder.split("/")[-1]
            
            
            # Check if any split contains the annotation names instead of just the first
            index = 0
            original_prefix = prefix
            while index < len(prefix_split):
                if folder_name + "_" + prefix_split[index] in label_dict:
                    prefix = folder_name + "_" + prefix
                    label_prefix = folder_name + "_" + prefix_split[index]
                    break

                prefix = prefix.replace(prefix_split[index]+"_","")
                index += 1
            
            if index == len(prefix_split):
                raise Exception(f"{original_prefix} cannot be found in annotations file.")

            try:
                df_raw_track = pd.read_csv(track_path, encoding='latin1',
                                        header=None)
                names_track = df_raw_track.iloc[0].tolist()
                df_track = pd.read_csv(track_path, encoding='latin1',
                                    skiprows=1, names=names_track)
                #all as float
                df_track = df_track.apply(pd.to_numeric, errors='coerce')

                df_track['PREFIX'] = prefix
                df_track['LABEL'] = label_dict[label_prefix]

                df_raw_spot = pd.read_csv(spot_path, encoding='latin1',
                                        header=None)
                names_spot = df_raw_spot.iloc[0].tolist()
                df_spot = pd.read_csv(spot_path, encoding='latin1',
                                    skiprows=1, names=names_spot)
                df_spot = df_spot.apply(pd.to_numeric, errors='coerce')
                df_spot['PREFIX'] = prefix
                df_spot['LABEL'] = label_dict[label_prefix]

                tracks.append(df_track)
                spots.append(df_spot)

            except Exception as e:
                import sys, traceback
                print(label_dict)
                print(f"Failed to load {prefix}: {e}")
                traceback.print_exc()
                sys.exit(1)
                
                continue

    spots_df = pd.concat(spots, ignore_index=True)
    tracks_df = pd.concat(tracks, ignore_index=True)

    return spots_df, tracks_df


# === Step 3: Filter Valid Trajectories ===
def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames
                          ][["PREFIX", "TRACK_ID"]]
    spots_df_filtered = spots_df.merge(valid_ids, 
                                       on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df_filtered = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]


    return spots_df_filtered, tracks_df_filtered

def compute_msd(x, y, max_lag=None):
    N = len(x)
    if max_lag is None:
        max_lag = N // 4
    msd = []
    for dt in range(1, max_lag + 1):
        dx = x[dt:] - x[:-dt]
        dy = y[dt:] - y[:-dt]
        squared_displacement = dx**2 + dy**2

        if len(squared_displacement) == 0:
            msd.append(0)
        else:
            msd.append(squared_displacement.mean())
    return msd

# === Step 4: Compute Features ===
def compute_features(spots_df):
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2
                                + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"],
                                       spots_df["VELOCITY_X"]) / np.pi
    
    spots_df["MEAN_SQUARE_DISPLACEMENT"] = 0
    all_msd = []

    total_tracks = len(spots_df.groupby(["PREFIX", "TRACK_ID"]))
    for (prefix, track_id), group in tqdm(spots_df.groupby(["PREFIX", "TRACK_ID"]),
                                          total = total_tracks,
                                          desc ="Calculating MSD"):

        group = group.sort_values("FRAME")
        x = group["POSITION_X"].values
        y = group["POSITION_Y"].values
        frames = group["FRAME"].values

        msd = compute_msd(x, y, max_lag=(len(x)-1))

        # align MSD with the correct FRAME (starting at lag = 1)
        lag_frames = group["FRAME"].iloc[1:].values
        for frame_val, m in zip(lag_frames, msd):
            all_msd.append({
                "PREFIX": prefix,
                "TRACK_ID": track_id,
                "FRAME": frame_val,
                "MEAN_SQUARE_DISPLACEMENT": m,
            })

    msd_df = pd.DataFrame(all_msd)

    # Add the calculated MSD column to spots_df.
    # There is now an intialized MSD and MSD_new
    spots_df = spots_df.merge(
        msd_df,
        on=["PREFIX", "TRACK_ID", "FRAME"],
        how="left",
        suffixes=("", "_new")
    )

    # Combine original, initialized MSD with new calculated MSD if possible, otherwise use the initialized value.
    # Done to avoid NaN when merging because calculated MSD does not contain MSD(0), resuting in NaN for the first entry when merging.
    spots_df["MEAN_SQUARE_DISPLACEMENT"] = spots_df["MEAN_SQUARE_DISPLACEMENT_new"].combine_first(
        spots_df["MEAN_SQUARE_DISPLACEMENT"]
    )
    spots_df = spots_df.drop(columns=["MEAN_SQUARE_DISPLACEMENT_new"])

    drop_cols = [col for col in spots_df.columns 
                 if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for f in features:
        spots_df[f] = pd.to_numeric(spots_df[f], errors='coerce')
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return spots_df

    
# === Step 5: Align and Save Sequences ===
def align_and_save_dataset(spots_df, features, seq_len=20,
                           output_prefix=""):
    X_list, y_list, track_id_list = [], [], []
    rows = []
    for (p, tid), traj in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        feat = traj[features].values
        if len(feat) >= seq_len:
            feat = feat[:seq_len]
        else:
            pad = np.zeros((seq_len - len(feat), len(features)))
            feat = np.vstack([feat, pad])

        feat_scaled = feat.copy()

        X_list.append(feat_scaled)
        y_list.append(traj["LABEL"].iloc[0])
        track_id_list.append((p, tid))

        for t in range(seq_len):
            row = [f"{p}_{tid}", t] + list(feat_scaled[t])
            rows.append(row)
    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)
    np.savez(f"{GENERATED_DIR}/{output_prefix}trajectory_dataset_{seq_len}.npz",
             X=X, y=y, track_ids=track_ids)

    df_out = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df_out.to_csv(f"{GENERATED_DIR}/{output_prefix}trajectory_dataset_{seq_len}.csv", index=False)

    print(
        f"[Save] Dataset saved: {GENERATED_DIR}/{output_prefix}trajectory_dataset_{seq_len}.npz & .csv "
        f"| Shape: {X.shape}"
    )



# === Step 6: Save Track-Level Dataset ===
def build_track_level_dataset(tracks_df, datasets,
                              output_prefix="", 
                              track_features = track_features):
    if len(track_features) == 0:
        print("[INFO] No track features available.")
        return
    
    merged_mapping = {}

    for label in datasets:
        overlap = merged_mapping.keys() & datasets[label].mapping.keys()  # intersection of keys
        if overlap:
            raise ValueError(f"Overlapping keys found when merging: {overlap}")
        merged_mapping.update(datasets[label].mapping)

    def match_label(prefix, merged_mapping):
        prefix = "_".join(prefix.split("_")[:2])
        return merged_mapping.get(prefix, np.nan)
    
    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label, args=(merged_mapping,))

    df = tracks_df.dropna(subset=track_features +
                          ["LABEL", "PREFIX", "TRACK_ID"]).copy()

    records = []
    for prefix, group in df.groupby("PREFIX"):
        group_scaled = group[track_features].values

        for i, row in enumerate(group.itertuples()):
            record = {
                "PREFIX": prefix,
                "TRACK_ID": row.TRACK_ID,
                "LABEL": row.LABEL
            }
            for j, f in enumerate(track_features):
                record[f] = group_scaled[i][j]
            records.append(record)

    df_final = pd.DataFrame(records)

    df_final.to_csv(f"{GENERATED_DIR}/{output_prefix}track_dataset.csv",
                    index=False)

    np.savez(f"{GENERATED_DIR}/{output_prefix}track_dataset.npz", 
            X=df_final[track_features].values, 
            y=df_final["LABEL"].values,
            track_ids=df_final[["PREFIX", "TRACK_ID"]].values)
    print(f"[Save] Dataset saved: {GENERATED_DIR}/{output_prefix}track_dataset.csv & .npz")


def filter_outer(spots_df):
    def log_debug_stats(df, conditions, label):
        total_rows = len(df)
        print(f"\n[Debug] --- {label} ---")
        print(f"[Debug] Current total rows: {total_rows}")

        for name, cond_fn in conditions.items():
            cond = cond_fn(df)   # Evaluate fresh on the current DataFrame
            print(f"[Debug] Rows with {name}: {cond.sum()}")

        return total_rows
        
    conditions_to_remove = {
        "ELLIPSE_MINOR == 0": lambda df: df['ELLIPSE_MINOR'] == 0,
        "ELLIPSE_ASPECTRATIO <= 0": lambda df: df['ELLIPSE_ASPECTRATIO'] <= 0,
        "ELLIPSE_ASPECTRATIO > 5": lambda df: df['ELLIPSE_ASPECTRATIO'] > 5,
    }

    # --------------------------------------
    # REMOVE ANY TRACK WHERE EVEN 1 INSTANCE 
    # DOES NOT FOLLOW THE ABOVE CONDITIONS
    # --------------------------------------

    init_rows = len(spots_df)
    condition_series = [cond(spots_df) for cond in conditions_to_remove.values()]
    rows_to_remove = pd.concat(condition_series, axis=1).any(axis=1)
    log_debug_stats(spots_df, conditions_to_remove, "Before Filtering")

    # -------------------------------------------
    # CLIP ANY OUTLIER VALUES TO A SPECIFIC VALUE
    # -------------------------------------------
    spots_df["ELLIPSE_ASPECTRATIO"] = spots_df["ELLIPSE_ASPECTRATIO"].clip(upper=5)

    # -------------------------------------------
    # REMOVE ANY OUTLIER GROUPS
    # -------------------------------------------
    bad_groups = spots_df.loc[rows_to_remove, ['PREFIX', 'TRACK_ID']].drop_duplicates()
    filtered_df = spots_df.merge(bad_groups, on=['PREFIX', 'TRACK_ID'], how='left', indicator=True)
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    spots_df = filtered_df
    
    log_debug_stats(spots_df, conditions_to_remove, "After Filtering")

    filtered_rows = len(spots_df)
    removed_count = init_rows - filtered_rows
    print(f"[Filter] Removed {removed_count} invalid rows | Remaining: {filtered_rows}")

    num_groups_removed = bad_groups.shape[0]
    print(f"Removed {num_groups_removed} cell tracks.")


    # --------------------------------------
    # REMOVE IF AVERAGE RADIUS SIZE IS TOO SMALL
    # --------------------------------------
    groups_before = spots_df.groupby(["PREFIX", "TRACK_ID"]).ngroups
    avg_radius = spots_df.groupby(["PREFIX", "TRACK_ID"])["RADIUS"].transform("mean")
    spots_df = spots_df[avg_radius >= 3].reset_index(drop=True)
    
    filtered_rows = len(spots_df)
    removed_count = init_rows - filtered_rows
    print(f"[Filter] Removed {removed_count} invalid rows because of RADIUS | Remaining: {filtered_rows}")
    groups_after = spots_df.groupby(["PREFIX", "TRACK_ID"]).ngroups

    num_groups_removed = groups_before - groups_after
    print(f"Removed {num_groups_removed} cell tracks.")

    return spots_df


def count_num_of_tracks(seq_path, track_path):
    from collections import defaultdict
    from Config import DATA_DIR
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
    test_label_value_dist_dict = defaultdict(int)

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            case_name = "_".join(tid[0].split("_")[:2])
            test_label_dist_dict[case_name] += 1
            test_label_value_dist_dict[y_seq[i]] += 1
    label_df = pd.DataFrame([test_label_dist_dict])
    label_df = label_df.T
    label_df["Label"] = [0.5, 0.5, 0.5, 0.5,
    1, 1, 0, 0,
    0.5, 0.5, 0.5, 0.5,
    1, 1, 1,
    0.5, 0.5, 0,
    0.5, 0, 1]

    label_df = label_df.reset_index()
    label_df.columns = ["Case Name", "Number of Tracks", "Label"]
    label_df = label_df.sort_values(by="Number of Tracks")
    label_df = label_df.sort_values(by="Label")

    label_df.to_csv(f"{DATA_DIR}/label counts.csv", index=False)
    print("Saved Label counts")


if __name__ == "__main__":
    datasets = {}

    for label in DATASET_CONFIGS:
        datasets[label] = Dataset_Batch(DATASET_CONFIGS[label]["annotation_path"], DATASET_CONFIGS[label]["data_folder"])
        datasets[label].mapping = load_annotations(datasets[label].annotation_path, datasets[label].data_folder)
    
    spots_df, tracks_df = load_tracks_and_spots(datasets)

    # Create train dataset
    spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
    spots_df = compute_features(spots_df)
    spots_df = filter_outer(spots_df)
    save_unscaled_spot_features(spots_df, output_prefix="")
    save_unscaled_track_features(tracks_df, datasets=datasets, output_prefix="")

    features = [ # Time-based Features 
        'AREA', 'PERIMETER', 'CIRCULARITY',
        'ELLIPSE_ASPECTRATIO','SOLIDITY', 
        'SPEED', "MEAN_SQUARE_DISPLACEMENT"
    ]

    from Config import SEQ_LEN
    for seq_len_iter in [SEQ_LEN]:
        align_and_save_dataset(spots_df,
                            features, seq_len=seq_len_iter,
                            output_prefix=SEQ_DATASET_PREFIX)
    
    build_track_level_dataset(tracks_df, datasets=datasets, output_prefix=TRACK_DATASET_PREFIX)   
    count_num_of_tracks(f"{GENERATED_DIR}/{SEQ_DATASET_PREFIX}trajectory_dataset_{SEQ_LEN}.npz",
                        f"{GENERATED_DIR}/{TRACK_DATASET_PREFIX}track_dataset.npz",)
    
    print("Dataset creation completed.")