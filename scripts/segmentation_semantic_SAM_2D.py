#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from skimage.io import imread
import tifffile as tf
import pyclesperanto_prototype as cle
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from skimage.measure import label
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

model_type = "vit_t"
sam_checkpoint = "/opt/T-MIDAS/models/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    return parser.parse_args()

args = parse_args()
    
image_folder = args.input

def process_image(image_path):
    try:

        image = imread(image_path)
        masks = mask_generator.generate(image) # generate masks using Mobile-SAM
        for i, mask_data in enumerate(masks):
              mask = mask_data["segmentation"]
              labeled_mask = label(mask, return_num=False)
              labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max())
        labels = cle.dilate_labels(labels, None, 1.0)
        labels = cle.merge_touching_labels(labels)
        labels = cle.pull(labels)

        return labels
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
    if not filename.endswith(".tif"):
        continue
    #print(f"Processing image: {filename}")
    labeled_image = process_image(os.path.join(image_folder, filename))
    if labeled_image is not None:
        tf.imwrite(os.path.join(image_folder, f"{filename[:-4]}_semantic_seg_sam.tif"), labeled_image, compression='zlib')
        
  