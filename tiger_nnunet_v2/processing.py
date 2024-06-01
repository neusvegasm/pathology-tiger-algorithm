import time
from functools import wraps
from pathlib import Path
from typing import List
import os

import numpy as np
from tqdm import tqdm

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage import measure
from skimage.measure import label, regionprops

from .gcio import (UNET_RAW_PATH, UNET_PREPROCESSED_PATH, UNET_RESULTS_PATH,
                   TMP_DETECTION_OUTPUT_PATH, TMP_SEGMENTATION_OUTPUT_PATH,
                   TMP_TILS_SCORE_PATH, copy_data_to_output_folders,
                   get_image_path_from_input_folder,
                   get_tissue_mask_path_from_input_folder,
                   initialize_output_folders)
from .rw import (READING_LEVEL, WRITING_TILE_SIZE, DetectionWriter,
                 SegmentationWriter, TilsScoreWriter,
                 open_multiresolutionimage_image)


# Set environment variables for nnUNet paths
os.environ["nnUNet_raw"] = str(UNET_RAW_PATH)
os.environ["nnUNet_preprocessed"] = str(UNET_PREPROCESSED_PATH)
os.environ["nnUNet_results"] = str(UNET_RESULTS_PATH)

# Decorator for timing function execution
# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap

def segmentation_to_bboxes(segmentation_mask, probability_map, threshold):
    """
    Extracts bounding boxes from a segmentation mask based on a probability threshold.

    Args:
        segmentation_mask (np.ndarray): The segmentation mask array.
        probability_map (np.ndarray): The probability map corresponding to each pixel in the segmentation mask.
        threshold (float): The minimum probability required to keep a bounding box.

    Returns:
        List[tuple]: A list of tuples, each representing a bounding box where the mean probability exceeds the threshold.
                     Each tuple contains (centroid_x, centroid_y, mean_prob).
    """
    labels = segmentation_mask == 2  # extract labels: background:0, wall: 1, cell:2
    labeled_mask = label(labels)
    # Measure properties of labeled regions, using the probability map as the intensity image
    props = regionprops(labeled_mask, intensity_image=probability_map)
    
    bboxes = []
    # Iterate over the properties of each labeled region
    for prop in props:
        # Get the centroid coordinates (y, x) of the region
        centroid_y, centroid_x = prop.centroid
        mean_prob = prop.mean_intensity

        # Filter bounding boxes based on the probability threshold
        if mean_prob > threshold:
            bboxes.append((float(centroid_x), float(centroid_y), float(mean_prob)))
    
    return bboxes

@timing
def process_image_tile_to_segmentation(
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray
) -> np.ndarray:
    """Example function that shows processing a tile from a multiresolution image for segmentation purposes.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        image_tile (np.ndarray): Input image tile.
        tissue_mask_tile (np.ndarray): Tissue mask tile.

    Returns:
        np.ndarray: Processed segmentation mask.
    """

    prediction = np.copy(image_tile[:, :, 0])
    prediction[image_tile[:, :, 0] > 90] = 1
    prediction[image_tile[:, :, 0] <= 90] = 2
    return prediction * tissue_mask_tile

@timing
def process_image_tile_to_detections(
    detection_writer,
    image_tiles,
    coordinates,
    threshold,
    spacing: tuple,
    predictor,
    num_cores
) -> List[tuple]:
    """Processes a batch of image tiles for detection purposes using nnUNet.

    Args:
        detection_writer (DetectionWriter): Writer to save detection results.
        image_tiles (List[np.ndarray]): List of image tiles.
        coordinates (List[tuple]): List of coordinates corresponding to each tile.
        threshold (float): Probability threshold for bounding box extraction.
        spacing (tuple): Spacing information for the tiles.
        predictor (nnUNetPredictor): nnUNet predictor instance.
        num_cores (int): Number of CPU cores to use for prediction.

    Returns:
        List[tuple]: List of tuples (x, y, probability) for each detection.
    """
    num_tiles = len(image_tiles)
     # Create a list of property dictionaries for each tile
    props = [{'spacing': (999, 1, 1)} for _ in range(num_tiles)]

    num_proc = int(num_cores/2)
    # Run the prediction on image tiles 
    predictions = predictor.predict_from_list_of_npy_arrays(image_tiles, None, props, None, num_processes=num_proc, save_probabilities=True)

    # Iterate over the predictions and corresponding coordinates to generate detections
    for (segmentation_map, probability_map), (x, y) in zip(predictions, coordinates):
        segmentation_map = segmentation_map[0]
        probability_map = probability_map[2][0]

        # Convert the segmentation map and probability map to bounding boxes using the threshold
        detections = segmentation_to_bboxes(segmentation_map, probability_map, threshold)

        detection_writer.write_detections(
            detections=detections, spacing=spacing, x_offset=x, y_offset=y
        )

@timing
def process_segmentation_detection_to_tils_score(
    segmentation_path: Path, detections: List[tuple]
) -> int:
    """Example function that shows processing a segmentation mask and corresponding detection for the computation of a TIL score.

    NOTE
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        segmentation_path (Path): Path to the segmentation mask file.
        detections (List[tuple]): List of detections.

    Returns:
        int: TIL score (between 0 and 100).
    """
    level = 4
    cell_area_level_1 = 16 * 16

    image = open_multiresolutionimage_image(path=segmentation_path)
    width, height = image.getDimensions()
    slide_at_level_4 = image.getUCharPatch(
        0, 0, int(width / 2 ** level), int(height / 2 ** level), level
    )
    area = len(np.where(slide_at_level_4 == 2)[0])
    cell_area = cell_area_level_1 // 2 ** 4
    n_detections = len(detections)
    if cell_area == 0 or n_detections == 0:
        return 0
    value = min(100, int(area / (n_detections / cell_area)))
    return value

def process():
    """Processes a test slide by segmenting it, detecting objects, and computing a TIL score."""
    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE  # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f"Processing image: {image_path}")
    print(f"Processing with mask: {tissue_mask_path}")

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    detection_threshold = 0.5
    tile_batch_size = 10000
    tile_iter = 0
    num_cores = os.cpu_count()
    
    print(f"Initialize nnUNet predictor")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    #load weights from pretrained model (trained on snellius)
    predictor.initialize_from_trained_model_folder(
        "/home/user/nnUnet_config/results",
        use_folds="all",
        checkpoint_name='checkpoint_best.pth',
    )

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    # Lists to store image tiles and their coordinates
    image_tiles = []
    coordinates = []

    print("Processing image...")
    # loop over image and get tiles along y and x axis
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            
            # Extract a patch from the tissue mask at the current (x, y) position
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()

            if not np.any(tissue_mask_tile):
                continue
            
            # Extract a patch from the image at the current (x, y) position
            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            )

            # segmentation step
            segmentation_mask = process_image_tile_to_segmentation(
                image_tile=image_tile, tissue_mask_tile=tissue_mask_tile
            )
            segmentation_writer.write_segmentation(tile=segmentation_mask, x=x, y=y)

            # Convert the image tile to the right shape
            # Move RGB(A) to front, add additional dim so that we have shape (1, 3, 512, 512)
            # which is default format for nnunet images
            image_tile = np.transpose(image_tile, (2, 0, 1))  # Move channels to the front
            image_tile = np.expand_dims(image_tile, axis=1)  # Add an additional dimension at the start

            image_tiles.append(image_tile)
            coordinates.append((x, y))

            if tile_iter % tile_batch_size == 0:
                # Process the current batch of image tiles for detections
                process_image_tile_to_detections(
                    detection_writer=detection_writer, 
                    image_tiles=image_tiles, 
                    coordinates=coordinates, 
                    threshold=detection_threshold, 
                    spacing=spacing, 
                    predictor=predictor, 
                    num_cores=num_cores
                )

                # Reset lists after processing a batch
                image_tiles = []
                coordinates = [] 

            tile_iter += 1

    # Process remaining image tiles
    process_image_tile_to_detections(
        detection_writer=detection_writer, 
        image_tiles=image_tiles, 
        coordinates=coordinates, 
        threshold=detection_threshold, 
        spacing=spacing, 
        predictor=predictor, 
        num_cores=num_cores
    )

    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print("Number of detections", len(detection_writer.detections))

    print("Compute tils score...")
    # compute tils score
    tils_score = process_segmentation_detection_to_tils_score(
        TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    )
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
