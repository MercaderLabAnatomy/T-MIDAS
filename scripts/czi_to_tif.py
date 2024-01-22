import os
import argparse
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


# parse arguments
parser = argparse.ArgumentParser(description='Process czi files.')
parser.add_argument('--input', type=str, help='path to the czi files')
args = parser.parse_args()

folder = args.input

def czi_scenes_to_tifs(filepath):
    # test using AICSImageIO
    aics_img = AICSImage(filepath, reconstruct_mosaic=True)
    # export each scene as tif
    for i in aics_img.scenes:
        aics_img.set_scene(i)
        print("Exporting " + str(i) + " with shape "+ str(aics_img.data.shape)+ " and dim order " + str(aics_img.dims.order) + " to ome.tiff.")
        OmeTiffWriter.save(aics_img.data, filepath.replace(".czi", f"_{i}.ome.tiff"))
        # #aics_img.save(filepath.replace(".czi", f"_{i}.ome.tiff"), select_scenes=[i]) # bug with colors?

for file in os.listdir(folder):
    if file.endswith(".czi"):
        czi_scenes_to_tifs(os.path.join(folder, file))
