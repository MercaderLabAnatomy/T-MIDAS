import argparse
from aicsimageio import AICSImage


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
        print(i)
        aics_img.save(filepath.replace(".czi", f"_scene_{i}.tif"), select_scenes=[i])

for file in os.listdir(folder):
    if file.endswith(".czi"):
        czi_scenes_to_tifs(os.path.join(folder, file))
