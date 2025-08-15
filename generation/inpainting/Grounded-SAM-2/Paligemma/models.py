from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def paligemma2_3b_vqav2(img_path):
    '''
    imgs_path : list of str
        List of paths to images
        i.e. ["path/to/image1.jpg", "path/to/image2.jpg"]
    '''
    model_id = "merve/paligemma2-3b-vqav2"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained("google/paligemma2-3b-pt-448")

    PROMPT = "What is one interesting object in the image that is neither too small to be noticeable nor so large that it occupies almost the entire frame?"
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    inputs = processor(image, PROMPT, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    decoded = processor.decode(output[0], skip_special_tokens=True)[len(PROMPT):]

    return decoded