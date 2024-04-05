# Script to get some regionprops of all objects in all tifs in a input_folder

import os
import argparse
import pandas as pd
from skimage import io
import cupy as cp
from cucim.skimage.measure import regionprops


# Argument Parsing
parser = argparse.ArgumentParser(description="Segments CLAHE images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
args = parser.parse_args()

input_folder = args.input



def get_regionprops(tif):
    # Read the tif
    img = cp.asarray(io.imread(tif))

    # Get the regionprops
    props = regionprops(img)

    # Get the properties
    df = pd.DataFrame()
    for i, prop in enumerate(props):
        df.loc[i, 'Filename'] = os.path.basename(tif)
        df.loc[i, 'Label'] = int(prop.label) 
        df.loc[i, 'Area'] = prop.area
        df.loc[i, 'Perimeter'] = prop.perimeter
        df.loc[i, 'Eccentricity'] = prop.eccentricity
        df.loc[i, 'MajorAxisLength'] = prop.major_axis_length
        df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length

    return df

def main():
    # Get the tifs
    tifs = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_labels.tif')]

    # Get the regionprops
    df = pd.DataFrame()
    for tif in tifs:
        df = pd.concat([df, get_regionprops(tif)]) 

    # Save the data in parent folder
    df.to_csv(os.path.join(input_folder, 'regionprops.csv'), index=False)

if __name__ == '__main__':
    main()
