import os
import subprocess
import argparse
import torch

from inference_propainter import inference_propainter

def process_video_subdirectories(directory_path, video_directory ,start_rank=0, end_rank=100):
    """
    Processes subdirectories in a given directory by calling a script for each subdirectory
    within the specified rank range.
    
    Args:
        directory_path (str): Path to the directory containing subdirectories.
        start_rank (int): Rank of the first subdirectory to process (default is 0).
        end_rank (int): Rank of the last subdirectory to process (default is 100).
    
    Returns:
        None
    """
    script_path = 'Propainter/inference_propainter.py'

    # List all entries in the directory
    entries = os.listdir(directory_path)
    
    # Filter for subdirectories and sort them
    subdirectories = sorted([d for d in entries if os.path.isdir(os.path.join(directory_path, d))])
    
    # Ensure the end rank is within the range of available files
    if end_rank == -1:
        end_rank = len(subdirectories)
    else:
        end_rank = min(end_rank, len(subdirectories))

    nb_errors = 0
    
    # Process each video file within the specified range
    for rank in range(start_rank, end_rank):
        subdirectory_path = os.path.join(directory_path, subdirectories[rank])
        masks_subdirectory_path = os.path.join(subdirectory_path, 'masks')
        video_name = subdirectories[rank]
        video_path = os.path.join(video_directory, video_name)+'.mp4'

        try:
            inference_propainter(video = video_path, mask = masks_subdirectory_path, output = subdirectory_path)
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA Out of Memory error while processing video {video_path}: {e}")
            torch.cuda.empty_cache()
            try:
                inference_propainter(video = video_path, mask = masks_subdirectory_path, output = subdirectory_path, resize_ratio=0.5)
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA Out of Memory error while processing video {video_path} with resize_ratio=0.5: {e}")
                torch.cuda.empty_cache()
                nb_errors += 1
                continue
            continue
        except Exception as e:
            print(f"Error processing video file {video_path}: {e}")
            nb_errors += 1
            continue
    
    print(f"Processed {end_rank - start_rank - nb_errors} video files from {video_directory}.")
    print(f"Masked saved in {directory_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory_path', type=str, required=True, help='Path to the directory containing videos_name_subdirectories.')
    parser.add_argument(
        '--video_directory', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument(
        '--start_rank', type=int, default=0, help='Rank of the first subdirectory to process. Default: 0.')
    parser.add_argument(
        '--end_rank', type=int, default=100, help='Rank of the last subdirectory to process. \'-1\' to select all videos in the folder. Default: 100.'
    )
    
    args = parser.parse_args()
    process_video_subdirectories(args.directory_path, args.video_directory, args.start_rank, args.end_rank)

# Example usage:
# process_video_subdirectories('/path/to/main/folder', 0, 10)
