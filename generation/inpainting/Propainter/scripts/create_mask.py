from PIL import Image
import os
import sys
import argparse
import random



def create_mask(h = None, w = None, name = 'mask.png', path = None):
    """
    Create a mask image of dimensions (h, w) with the top-left corner masked,
    and save it in a sibling directory.

    Parameters:
        h (int): Height of the mask.
        w (int): Width of the mask.
        name (str): Name of the mask file.
        path (str): Path to the image or video file.

    Returns:
        None: Saves the mask as 'mask.png' in the sibling directory.
    """
    # Get the dimensions of the image from the metadata
    from get_metadata import get_metadata

    if path :
        metadata = get_metadata(path)
        if metadata:
            h = metadata['Height']
            w = metadata['Width']
            parent_folder = os.path.dirname(path)
            name = 'mask' + '_' + os.path.split(parent_folder)[1]+".png"
        else:
            print("Error: Could not retrieve metadata from the file.")
            return

    # Create a new image with a black background in RGB
    mask = Image.new("RGB", (w, h), (0, 0, 0))

    # Define the size of the masked area (top-left corner)
    masked_width = w // 4  # Width of the masked area
    masked_height = h // 4  # Height of the masked area

    #select at random the position of the top left corner of the mask
    x_top_left = random.randint(0, w - masked_width)
    y_top_left = random.randint(0, h - masked_height)

    # Create a white rectangle for the masked area
    for y in range(y_top_left, y_top_left + masked_height):
        for x in range(x_top_left, x_top_left + masked_width):
            mask.putpixel((x, y), (255, 255, 255))  # Set the masked area to white

    directory = 'mask'
    os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the mask as a PNG file in the sibling directory
    mask.save(os.path.join(directory, name), "PNG")
    print(f"Mask saved as '{directory}/{name}' with dimensions ({h}, {w}).")

# Code principal pour lire l'argument de ligne de commande
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--height', type=int, default = None , help='Height of the mask')
    parser.add_argument(
        '--width', type=int, default= None, help='Width of the mask')
    parser.add_argument(
        '-n', '--name', type=str, default='mask.png', help='Name of the mask file')
    parser.add_argument(
        '-p', '--path', type=str, default = None, help='Path to the image or video file')
    
    args = parser.parse_args()

    create_mask(args.height, args.width, args.name, args.path)