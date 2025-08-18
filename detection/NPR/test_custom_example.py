#!/usr/bin/env python3
"""
Example script demonstrating how to test the deepfake detection model on custom datasets.

This script shows different ways to test your model on any type of dataset folder.
"""

import os
import sys
import subprocess

def run_test_command(command):
    """Run a test command and print the output."""
    print(f"Running: {command}")
    print("-" * 50)
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Error running command: {e}")
    print("-" * 50)

def main():
    # Example 1: Test on a custom dataset with default settings
    print("Example 1: Testing on a custom dataset with default settings")
    custom_dataset_path = "/path/to/your/custom/dataset"
    model_path = "NPR.pth"
    
    command1 = f"CUDA_VISIBLE_DEVICES=0 python test.py --model_path {model_path} --batch_size 32 --custom_dataset {custom_dataset_path}"
    run_test_command(command1)
    
    # Example 2: Test on a custom dataset with no resizing (for datasets with consistent image sizes)
    print("\nExample 2: Testing on a custom dataset with no resizing")
    command2 = f"CUDA_VISIBLE_DEVICES=0 python test.py --model_path {model_path} --batch_size 32 --custom_dataset {custom_dataset_path} --no_resize"
    run_test_command(command2)
    
    # Example 3: Test on a custom dataset with cropping enabled
    print("\nExample 3: Testing on a custom dataset with cropping enabled")
    command3 = f"CUDA_VISIBLE_DEVICES=0 python test.py --model_path {model_path} --batch_size 32 --custom_dataset {custom_dataset_path} --no_crop"
    run_test_command(command3)
    
    # Example 4: Test both custom dataset and all predefined datasets
    print("\nExample 4: Testing both custom dataset and all predefined datasets")
    command4 = f"CUDA_VISIBLE_DEVICES=0 python test.py --model_path {model_path} --batch_size 32 --custom_dataset {custom_dataset_path} --test_all"
    run_test_command(command4)

if __name__ == "__main__":
    print("Custom Dataset Testing Examples")
    print("=" * 50)
    print("Note: Please modify the dataset path and model path in this script before running.")
    print("=" * 50)
    
    # Check if user wants to run examples
    response = input("Do you want to run the example commands? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Examples not run. You can modify the script and run it manually.") 