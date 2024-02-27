import os
import argparse
import numpy as np
import tifffile as tf
from readlif.reader import LifFile

# parse arguments
parser = argparse.ArgumentParser(description='Process a lif file.')
parser.add_argument('--input', type=str, help='path to the lif file')
args = parser.parse_args()

input_folder = args.input



output_dir = input_folder + "/tif_files"

# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#input_folder = "/media/geffjoldblum/DATA/Romario"

lif_files = []
for file in os.listdir(input_folder):
    if file.endswith(".lif"):
        lif_files.append(file)





def scene_to_stack(scene):
    n_channels = scene.info['channels']
    array_list = []
    for i in range(n_channels):
        # get the array of the current scene
        frames = []
        for j in range(scene.nz):
        # get the frame
            frame = scene.get_frame(t=0,c=i,z=j)
            # convert frame to numpy array
            frame = np.array(frame)
            frames.append(frame)
        frames = np.array(frames)
        array_list.append(frames)

    multichannel_stack = np.array(array_list)
    return multichannel_stack

def create_metadata(scene):
    # get resolution
    scale_x = scene.info['scale_n'][1]
    scale_y = scene.info['scale_n'][2]
    
    # check if scale_z is defined in the dictionary
    if 3 in scene.info['scale_n']:
        scale_z = scene.info['scale_n'][3]
    else:
        scale_z = 1
    
    # get resolution in um
    x_res = 1 / scale_x 
    y_res = 1 / scale_y 
    z_res = 1 / scale_z 

    n_channels = scene.info['channels']
    
    metadata = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <Pixels ID="Pixels:0" SizeX="{sizex}" SizeY="{sizey}" SizeC="{sizec}" SizeT="{sizet}" SizeZ="{sizez}" PhysicalSizeX="{psizex}" PhysicalSizeY="{psizey}" PhysicalSizeZ="{psizez}" PhysicalSizeXUnit="{psizexu}" PhysicalSizeYUnit="{psizeyu}" PhysicalSizeZUnit="{psizezu}" DimensionOrder="{dimorder}">
            </Pixels>
        </Image>
    </OME>
    """

    metadata = metadata.format(sizec=scene.info['dims'][4], sizet=scene.info['dims'][3], sizez=scene.info['dims'][2], sizex=scene.info['dims'][0], sizey=scene.info['dims'][1], psizex=x_res, psizey=y_res, psizez=z_res, psizexu='um', psizeyu='um', psizezu='um', dimorder='TZYXC')

    # convert metadata to 7bit ascii
    metadata = metadata.encode('ascii', errors='ignore')
    #print(metadata)
    return metadata

def save_image(image,metadata,path):
    # save the image stack
    tf.imwrite(path, image,description=metadata)

def process_scene(scene,path):
    multichannel_stack = scene_to_stack(scene)
    # create metadata
    metadata = create_metadata(scene)
    position = scene.info['name']
    # replace "/" with "_" in position
    position = position.replace("/","_")
    # save the multichannel stack
    save_image(multichannel_stack,metadata,path.split(".")[0] +"_{scene}.tif".format(scene=position))
    print("Processed {scene}".format(scene=position))

for lif_file in lif_files:
    
    if lif_file:
        
        path = os.path.join(input_folder, lif_file)
        # read the lif file
        lif = LifFile(path)
        # index all scenes in the lif file
        img_list = [i for i in lif.get_iter_image()]

        for scene in img_list:
            process_scene(scene,path)    

    else:
        print("No lif file found in the input folder.")
        exit()











