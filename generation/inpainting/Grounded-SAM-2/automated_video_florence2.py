import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

from  automated_florence2_scripts import object_detection_and_segmentation

"""
Hyperparam for Ground and Tracking
"""

def glob_function(video_path, save_tracking_results_dir, output_video_path=None, prompt_type_for_video = 'box', number = 1, merge = True):

    video_name = os.path.split(VIDEO_PATH)[1]
    VIDEO_PATH = video_path
    SAVE_TRACKING_RESULTS_DIR = os.path.join(save_tracking_results_dir, video_name)
    OUTPUT_VIDEO_PATH = output_video_path
    PROMPT_TYPE_FOR_VIDEO = prompt_type_for_video


    """
    Step 1: Environment settings and model initialization for SAM 2
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    """
    Custom video input directly using video files
    """
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

    if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(SAVE_TRACKING_RESULTS_DIR)

    SOURCE_VIDEO_FRAME_DIR = os.path.join(SAVE_TRACKING_RESULTS_DIR, "frames")
    if not os.path.exists(SOURCE_VIDEO_FRAME_DIR):
        os.makedirs(SOURCE_VIDEO_FRAME_DIR)

    # Create a directory to save the masks
    SAVE_MASKS_DIR = os.path.join(SAVE_TRACKING_RESULTS_DIR, "masks")
    if not os.path.exists(SAVE_MASKS_DIR):
        os.makedirs(SAVE_MASKS_DIR)

    # saving video to frames
    source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Florence2 for the first bbox that we pass to SAM2 image segmenter
    """

    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    input_boxes, class_names = object_detection_and_segmentation(
            image_path=img_path)
    
    input_boxes = input_boxes
    print("class_names")

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    OBJECTS = class_names

    print(OBJECTS)

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the mask shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """

    for frame_idx, segments in video_segments.items():
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        # Save each mask as a PNG file

        #Check if we want to merge masks
        if merge == 'True':
            # Create an empty mask for merging
            merged_mask = np.zeros_like(masks[0], dtype=np.uint8)
    
            # Merge all masks
            for mask in masks:
                merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8)
            merged_mask = (merged_mask * 255).astype(np.uint8)
            merged_mask_filename = os.path.join(SAVE_MASKS_DIR, f"frame_{frame_idx:05d}_merged.png")
            cv2.imwrite(merged_mask_filename, merged_mask)

        else:
            for obj_id, mask in zip(object_ids, masks):
                mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8
                mask_filename = os.path.join(SAVE_MASKS_DIR, f"frame_{frame_idx:05d}_object_{obj_id}.png")
                cv2.imwrite(mask_filename, mask)

    """
    Step 6: Convert the annotated frames to video
    """

    #create_video_from_images(SOURCE_VIDEO_FRAME_DIR, OUTPUT_VIDEO_PATH)

    #print(f"Tracking video saved at {OUTPUT_VIDEO_PATH}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--video_path', type=str, help='Path of the video .mp4')
    parser.add_argument(
        '-o', '--output_video_path', type=str, default="", help='Path to save the tracking video.')
    parser.add_argument(
        '-p', '--prompt_type_for_video', choices=['point', 'box', 'mask'], default='box', help='Prompt type for video predictor. Choose from ["point", "box", "mask"]')
    parser.add_argument(
        '-s', '--save_tracking_results_dir', type=str, default='./tracking_results', help='Directory path to save tracking results. Default = "./tracking_results"')
    parser.add_argument(
        '--number', type=int, default=1, help='Maximum number of masks, \'-1\' for all masks, Default = 1')
    parser.add_argument(
        '--merge', choices=['True','False'], default='True', help='Merge masks, Default = True')
    

    args = parser.parse_args()
    video_path = args.video_path
    output_video_path = args.output_video_path
    prompt_type_for_video = args.prompt_type_for_video
    save_tracking_results_dir = args.save_tracking_results_dir 


    number = args.number
    merge = args.merge

    glob_function(video_path, save_tracking_results_dir, output_video_path, prompt_type_for_video, number, merge)
    