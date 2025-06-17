"""
Remove Audio from Videos Recursively
===================================

Author
------
Gaëtan Brison

Description
-----------
This script recursively scans a directory tree for `.mp4` files and removes their audio tracks,
replacing each file in-place using `ffmpeg`.

Usage
-----
Run the script via the command line:

    python remove_audio.py --input_dir random/

Dependencies
------------
- ffmpeg: Must be installed and accessible in your system PATH

Functions
---------
- find_ffmpeg(): Locates ffmpeg binary in PATH.
- parse_args(): Parses command-line arguments.
- remove_audio(): Invokes ffmpeg to remove audio from a given file.
- main(): Entry point that applies audio removal to all matching files.
"""

import os
import sys
import argparse
import subprocess

def find_ffmpeg():
    """Locate the ffmpeg binary.

    Returns
    -------
    str
        Path to the ffmpeg executable.

    Raises
    ------
    SystemExit
        If ffmpeg is not found in PATH.
    """
    from shutil import which
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    sys.exit("ERROR: ffmpeg not found in PATH.")

def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with input_dir attribute.
    """
    parser = argparse.ArgumentParser(description="Remove audio from all videos recursively.")
    parser.add_argument("--input_dir", required=True, help="Root directory to scan for videos")
    return parser.parse_args()

def remove_audio(ffmpeg_bin, input_path, output_path):
    """Remove audio from a single video file using ffmpeg.

    Parameters
    ----------
    ffmpeg_bin : str
        Path to the ffmpeg binary.
    input_path : str
        Path to the input .mp4 file.
    output_path : str
        Path to the temporary output file with audio removed.
    """
    cmd = [
        ffmpeg_bin,
        "-y",            # Overwrite output file if exists
        "-i", input_path,
        "-c:v", "copy",  # Copy video stream as-is
        "-an",           # Remove audio stream
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"\u2713 Removed audio: {input_path}")
    except subprocess.CalledProcessError:
        print(f"\u2717 Failed to process: {input_path}")

def main():
    """Main function that removes audio from all .mp4 files in the given directory."""
    args = parse_args()
    ffmpeg_bin = find_ffmpeg()

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                input_path = os.path.join(root, file)
                output_path = input_path  # Overwrite original file

                # Temporary output file to avoid conflict during write
                temp_output = input_path + ".noaudio.mp4"
                remove_audio(ffmpeg_bin, input_path, temp_output)

                # Replace original with audio-stripped version
                os.replace(temp_output, input_path)

if __name__ == "__main__":
    main()
