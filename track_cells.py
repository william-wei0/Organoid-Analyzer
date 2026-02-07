import os
import imagej
import jpype
import scyjava
from Config import DATA_DIR, FIJI_PATH, JAVA_ARGUMENTS, DATASET_CONFIGS
import pandas as pd

def track_with_trackmate(images_folder, subcase_names, case_name, fiji_path, prefix = "", specific_image_transformations ={}, 
                         include_spots_without_track_id = False, ignore_duplicate_warning = False, java_arguments = ""):

    # Check if output folder can be created
    output_folder_path = os.path.join(DATA_DIR, case_name)
    if os.path.exists(output_folder_path) and not ignore_duplicate_warning:
        user_input = ""

        while user_input.lower() != "y" or user_input.lower() != "n":
            user_input = input(f"\n\nA folder with case name: '{case_name}' already exists at: '{output_folder_path}' \n" +
                        f"Should the current run replace the previous results in the folder? (y/n): ")
    
            if user_input == "y":
                import shutil
                try:
                    shutil.rmtree(output_folder_path)
                    print(f"Folder '{output_folder_path}' and its contents deleted successfully.")
                except OSError as e:
                    print(f"Error: {e}")
                break
            elif user_input == "n":
                print("\nExiting current tracking run. Please change the case name.")
                return

    os.makedirs(output_folder_path, exist_ok=True)

    # -------------------------------------------------------------------
    # Launch Fiji with TrackMate
    # -------------------------------------------------------------------

    scyjava.config.add_options(java_arguments)
    ij = imagej.init(fiji_path, mode="interactive")
    print(ij.getApp().getInfo(True))

    # Import Java basic classes
    Model = jpype.JClass('fiji.plugin.trackmate.Model')
    TrackMate = jpype.JClass('fiji.plugin.trackmate.TrackMate')
    ThresholdDetectorFactory = jpype.JClass('fiji.plugin.trackmate.detection.ThresholdDetectorFactory')
    SimpleSparseLAPTrackerFactory = jpype.JClass('fiji.plugin.trackmate.tracking.jaqaman.SimpleSparseLAPTrackerFactory')
    TMUtils = jpype.JClass("fiji.plugin.trackmate.util.TMUtils")
    DetectionUtils = jpype.JClass("fiji.plugin.trackmate.detection.DetectionUtils")
    MaskUtils = jpype.JClass("fiji.plugin.trackmate.detection.MaskUtils")
    Integer = jpype.JClass("java.lang.Integer")
    Double = jpype.JClass("java.lang.Double")
    JDouble_array = jpype.JArray(jpype.JDouble, 1)
    
    for XY_position_folder_path in sorted(os.listdir(images_folder)):
        XY_position_folder_path = os.path.join(images_folder, XY_position_folder_path)
        if not os.path.isdir(XY_position_folder_path):
            continue  # skip files at root level
    
        all_tracks = []
        all_spots = []

        if os.path.exists(XY_position_folder_path):
            print(f"Processing {XY_position_folder_path} ...")
        else:
            print(f"FILEDNE {XY_position_folder_path}")
            return
        
        # Load image
        FolderOpener = jpype.JClass("ij.plugin.FolderOpener")
        imp = FolderOpener.open(XY_position_folder_path)

        # Set calibration (equivalent to Analyze > Set Scale...)
        cal = imp.getCalibration()
        cal.pixelWidth  = 0.33   # µm per pixel in X
        cal.pixelHeight = 0.33   # µm per pixel in Y
        cal.setUnit("µm")         # unit to display
        cal.setTimeUnit("frames")
        imp.setCalibration(cal)

        # Apply transformations
        for transformation in specific_image_transformations:
            for image_path in specific_image_transformations[transformation]:
                if (image_path in XY_position_folder_path and
                    type(specific_image_transformations[transformation]) != dict):
                    ij.IJ.run(imp, transformation, "stack")

        # Convert Z slices to T slices
        HyperStackConverter = jpype.JClass("ij.plugin.HyperStackConverter")
        nframes = imp.getStackSize()
        imp = HyperStackConverter.toHyperStack(imp, 1, 1, nframes, "xyctz", "composite")    

        # Initialize Trackmate Model
        model = Model()
        model.setPhysicalUnits( cal.getUnit(), cal.getTimeUnit() )
        settings = jpype.JClass("fiji.plugin.trackmate.Settings")(imp)
        print("Space units:", model.getSpaceUnits())
        print("Time units:", model.getTimeUnits())

        # Calculate Threshold using Otsu
        channel, t = 0, 0 
        img = TMUtils.rawWraps(imp)
        im_frame = DetectionUtils.prepareFrameImg(img, channel, t)
        interval = TMUtils.getInterval(img, settings)  
        interval = DetectionUtils.squeeze(interval)
        intensity_threshold = MaskUtils.otsuThreshold(im_frame)

        # Sometimes Auto-threshold gets it wrong, and detects too many cells
        # use SPECIAL_THRESHOLDING in Config to set a specific threshold for a 
        # specific folder
        for image_path in specific_image_transformations.get("specific_thresholds", {}):
            if image_path in XY_position_folder_path:
                intensity_threshold = Double(specific_image_transformations["specific_thresholds"][image_path])

        print("Using intensity threshold of: ", intensity_threshold)

        # Detector Settings
        settings.detectorFactory = ThresholdDetectorFactory()
        settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
        settings.detectorSettings = {
            'INTENSITY_THRESHOLD': intensity_threshold,  # auto threshold
            'TARGET_CHANNEL': Integer(1),
            'SIMPLIFY_CONTOURS': True,
        }
        
        # Tracker Settings
        settings.trackerFactory = SimpleSparseLAPTrackerFactory()
        settings.trackerSettings = settings.trackerFactory.getDefaultSettings() # almost good enough
        settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
        settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
        settings.trackerSettings['LINKING_MAX_DISTANCE'] = 15.0
        settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 15.0
        settings.trackerSettings.put("MAX_FRAME_GAP", Integer(2))
        settings.tstart = 0
        settings.tend = imp.getNFrames() - 1
        settings.dt = 1.0

        # Get spot analyzer factories
        SpotFitEllipseAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotFitEllipseAnalyzerFactory')
        SpotIntensityMultiCAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotIntensityMultiCAnalyzerFactory')
        SpotShapeAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotShapeAnalyzerFactory')
        SpotContrastAndSNRAnalyzerFactory = jpype.JClass('fiji.plugin.trackmate.features.spot.SpotContrastAndSNRAnalyzerFactory')

        settings.addSpotAnalyzerFactory(SpotFitEllipseAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotIntensityMultiCAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotShapeAnalyzerFactory())
        settings.addSpotAnalyzerFactory(SpotContrastAndSNRAnalyzerFactory())

        # Get track analyzers
        TrackBranchingAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackBranchingAnalyzer')
        TrackDurationAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackDurationAnalyzer')
        TrackIndexAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackIndexAnalyzer')
        TrackLocationAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackLocationAnalyzer')
        TrackMotilityAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackMotilityAnalyzer')
        TrackSpeedStatisticsAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer')
        TrackSpotQualityFeatureAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.track.TrackSpotQualityFeatureAnalyzer')
        DirectionalChangeAnalyzer = jpype.JClass('fiji.plugin.trackmate.features.edges.DirectionalChangeAnalyzer')
        # Edge Analyzer is needed to calculate mean directional change rate
        
        settings.addTrackAnalyzer(TrackBranchingAnalyzer())
        settings.addTrackAnalyzer(TrackDurationAnalyzer())
        settings.addTrackAnalyzer(TrackIndexAnalyzer())
        settings.addTrackAnalyzer(TrackLocationAnalyzer())
        settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())
        settings.addTrackAnalyzer(TrackSpotQualityFeatureAnalyzer())
        settings.addTrackAnalyzer(TrackMotilityAnalyzer())
        settings.addEdgeAnalyzer(DirectionalChangeAnalyzer())

        # ---------------------------------------------------------------------------------------------------
        # RUN DETECTOR AND COMPUTER SPOT FEATURES TO CALCULATE INITAL THRESHOLDING VALUE (QUALTIY THRESHOLD)
        # ---------------------------------------------------------------------------------------------------
        trackmate = TrackMate(model, settings)

        if not trackmate.checkInput():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Invalid input. {XY_position_folder_path}")
            continue

        if not trackmate.execDetection():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Unable to process tracking. {XY_position_folder_path}")
            continue
        if not trackmate.execInitialSpotFiltering():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Unable to process tracking. {XY_position_folder_path}")
            continue
        if not trackmate.computeSpotFeatures(True):
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Unable to process tracking. {XY_position_folder_path}")
            continue
        
        # -------------------------------------------------------------
        # AUTO CALCULATE INITAL THRESHOLD (a.k.a. QUALITY THRESHOLD)
        # -------------------------------------------------------------

        spot_collection = model.getSpots()
        qualities = [spot.getFeature("QUALITY") for spot in spot_collection.iterable(True) if spot.getFeature("QUALITY") is not None]

        if not qualities:
            raise ValueError("No spots detected, cannot compute threshold")
        qualities_array = JDouble_array(qualities)

        inital_quality_threshold = float(TMUtils.otsuThreshold(qualities_array))
        print("Inital threshold:", inital_quality_threshold)
        settings.initialSpotFilterValue = (Double(inital_quality_threshold))

        # -------------------------------------------------------------------------------
        # RUN TRACKMATE FULLY USING THE CALCULATED INITAL THRESHOLDING VALUE (QUALITY THRESHOLD)
        # using trackmate.process()
        # -------------------------------------------------------------------------------

        trackmate = TrackMate(model, settings)
        
        if not trackmate.process():
            print(str(trackmate.getErrorMessage()))
            print(f"TrackMate: Unable to process tracking. {XY_position_folder_path}")
            continue

        spot_collection = model.getSpots()
        feature_model = model.getFeatureModel()
        track_model = model.getTrackModel()

        
        # -------------------------------------------------------------------------------
        # GENERATE TRACK RESULTS DATAFRAME
        # -------------------------------------------------------------------------------
        for track_id in track_model.trackIDs(True):
            row = {
                "LABEL": f"Track_{track_id}",
            }

            for feature in [
                "TRACK_ID", "TRACK_INDEX", "NUMBER_SPOTS","NUMBER_GAPS","NUMBER_SPLITS","NUMBER_MERGES","NUMBER_COMPLEX",
                "LONGEST_GAP","TRACK_DURATION","TRACK_START","TRACK_STOP",
                "TRACK_DISPLACEMENT","TRACK_X_LOCATION","TRACK_Y_LOCATION","TRACK_Z_LOCATION",
                "TRACK_MEAN_SPEED","TRACK_MAX_SPEED","TRACK_MIN_SPEED","TRACK_MEDIAN_SPEED","TRACK_STD_SPEED",
                "TRACK_MEAN_QUALITY","TOTAL_DISTANCE_TRAVELED","MAX_DISTANCE_TRAVELED",
                "CONFINEMENT_RATIO","MEAN_STRAIGHT_LINE_SPEED","LINEARITY_OF_FORWARD_PROGRESSION",
                "MEAN_DIRECTIONAL_CHANGE_RATE"
            ]:
                val = feature_model.getTrackFeature(track_id, feature)
                row[feature] = float(val) if val is not None else None

            all_tracks.append(row)

        # -------------------------------------------------------------------------------
        # GENERATE SPOTS RESULTS DATAFRAME
        # -------------------------------------------------------------------------------
        for spot in spot_collection.iterable(True):
            track_id = model.getTrackModel().trackIDOf(spot)

            if include_spots_without_track_id or track_id is not None:
                row = {
                    "LABEL": spot.getName(),
                    "ID": spot.ID(),
                    "TRACK_ID": track_id,
                }

                for feature in [
                    "QUALITY","POSITION_X","POSITION_Y","POSITION_Z","POSITION_T","FRAME",
                    "RADIUS","VISIBILITY","MANUAL_SPOT_COLOR",
                    "MEAN_INTENSITY_CH1","MEDIAN_INTENSITY_CH1","MIN_INTENSITY_CH1","MAX_INTENSITY_CH1",
                    "TOTAL_INTENSITY_CH1","STD_INTENSITY_CH1","CONTRAST_CH1","SNR_CH1",
                    "ELLIPSE_X0","ELLIPSE_Y0","ELLIPSE_MAJOR","ELLIPSE_MINOR","ELLIPSE_THETA",
                    "ELLIPSE_ASPECTRATIO","AREA","PERIMETER","CIRCULARITY","SOLIDITY","SHAPE_INDEX"
                ]:
                    val = spot.getFeature(feature)
                    row[feature] = float(val) if val is not None else None
                all_spots.append(row)

        # -------------------------------------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------------------------------------
        print(f"Saving Results ...")
        df_spots = pd.DataFrame(all_spots)
        df_spots = df_spots.sort_values(by="TRACK_ID")

        df_tracks = pd.DataFrame(all_tracks)
        df_tracks = df_tracks.sort_values(by="TRACK_ID")
        subcase_folder_path = XY_position_folder_path.split(os.sep)[-2]
        XY_position = XY_position_folder_path.split(os.sep)[-1]

        for subcase_name in subcase_names:
            if subcase_name in subcase_folder_path:
                spots_output_path = os.path.join(output_folder_path, f"{prefix}{subcase_name}_{XY_position}_spots.csv")
                track_output_path = os.path.join(output_folder_path, f"{prefix}{subcase_name}_{XY_position}_tracks.csv")

                df_spots.to_csv(spots_output_path, index=False)
                df_tracks.to_csv(track_output_path, index=False)
                print("Done! Exported all track and spot features to CSV at: ")
                print(f"'{spots_output_path}' and '{track_output_path}'\n")
                break
        else:
            print(f"ERROR: {subcase_folder_path} is not associated with any {subcase_names}.")
            print(f"ERROR: Please update {subcase_names} or rename '{subcase_folder_path}' to contain one of the subcase names.")
            print(f"ERROR: Results will not be saved.")
            input("Press Enter to continue.")

if __name__ == "__main__":
    for case in DATASET_CONFIGS:
        for file in sorted(os.listdir(DATASET_CONFIGS[case]["images_folder"])):
            images_subfolder = os.path.join(DATASET_CONFIGS[case]["images_folder"], file)
            if os.path.isdir(images_subfolder):
                track_with_trackmate(images_subfolder, 
                                    DATASET_CONFIGS[case]["subcase_names"],
                                    DATASET_CONFIGS[case]["case_name"], 
                                    FIJI_PATH, 
                                    prefix = DATASET_CONFIGS[case]["prefix"],
                                    specific_image_transformations = DATASET_CONFIGS[case].get("specific_image_transformations", {}),
                                    java_arguments = JAVA_ARGUMENTS,
                                    include_spots_without_track_id = False, 
                                    ignore_duplicate_warning = True)
    print("------------------------")
    print("Cell Tracking Complete!")
    print("------------------------")