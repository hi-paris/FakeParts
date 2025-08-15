                  
import os
import argparse

from automated_video_paligemma2_v1_scripts import glob_function

def process_videos_in_range(folder_path, saving_dir, start_rank=0, end_rank=100):
    """
    Processes video files in a given folder by calling fonction1 on each video within a specified rank range.
    
    Parameters:
    folder_path (str): Path to the folder containing video files.
    start_rank (int, optional): The starting rank of the video to process (default is 0).
    end_rank (int, optional): The ending rank of the video to process (default is 100).
    """
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter and sort files to include only video files (e.g., .mp4, .wav)
    video_files = sorted([f for f in files if f.lower().endswith(('.mp4', '.wav'))])
    
    # Ensure the end rank is within the range of available files
    if end_rank == -1:
        end_rank = len(video_files)
    else:
        end_rank = min(end_rank, len(video_files))

    nb_errors = 0
    
    # Process each video file within the specified range
    for rank in range(start_rank, end_rank):
        video_path = os.path.join(folder_path, video_files[rank])

        try:
            glob_function(video_path, saving_dir)
        except Exception as e:
            print(f"Error processing video file {video_path}: {e}")
            nb_errors += 1
            continue

    print(f"Processed {end_rank - start_rank - nb_errors} video files from {folder_path}.")
    print(f"Masked saved in {saving_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_dir', type=str, required=True, help='Path to the directory containing video files.')
    parser.add_argument(
        '--saving_dir', type=str, required=True, help='Path to the directory to save the masks.')
    parser.add_argument(
        '--start_rank', type=int, default=0, help='The starting rank of the video to process. Default: 0.')
    parser.add_argument(
        '--end_rank', type=int, default=100, help='The ending rank of the video to process. \'-1\' to select all videos in the folder. Default: 100.'
    )
    
    args = parser.parse_args()
    process_videos_in_range(args.video_dir, args.saving_dir, args.start_rank, args.end_rank)