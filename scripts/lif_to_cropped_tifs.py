from readlif.reader import LifFile
import numpy as np
import cv2
import pyclesperanto_prototype as cle
import tifffile as tf
import argparse


# parse arguments
parser = argparse.ArgumentParser(description='Process a lif file.')
parser.add_argument('--lif', type=str, help='path to the lif file')
parser.add_argument('--template_channel', type=int, help='channel to use as template')
args = parser.parse_args()

if args.lif:
    img = LifFile(args.lif)
else:
    print("Please provide a path to the lif file.")
    exit()


# index all series in the lif file
img_list = [i for i in img.get_iter_image()]

if args.template_channel:
    template_channel = args.template_channel
else:
    print("Please provide a channel to use as template.")
    exit()





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

# scene_to_stack(img_list[1]).shape

#scene = img_list[2]

def create_metadata(scene):
    # get resolution
    scale_x = scene.info['scale_n'][1]
    scale_y = scene.info['scale_n'][2]
    # if scale_z is not defined, set it to 1
    if scene.info['scale_n'][3] == None:
        scale_z = 1
    else:
        scale_z = scene.info['scale_n'][3]
    # get resolution in um
    x_res = 1/scale_x 
    y_res = 1/scale_y 
    z_res = 1/scale_z 

    n_channels = scene.info['channels']
    metadata = """
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <Pixels ID="Pixels:0" SizeX="{sizex}" SizeY="{sizey}" SizeC="{sizec}" SizeT="{sizet}" SizeZ="{sizez}" PhysicalSizeX="{psizex}" PhysicalSizeY="{psizey}" PhysicalSizeZ="{psizez}" PhysicalSizeXUnit="{psizexu}" PhysicalSizeYUnit="{psizeyu}" PhysicalSizeZUnit="{psizezu}" DimensionOrder="{dimorder}">
            </Pixels>
        </Image>
    </OME>
    """

    metadata = metadata.format(sizec=scene.info['dims'][4],sizet=scene.info['dims'][3],sizez=scene.info['dims'][2],sizex=scene.info['dims'][0],sizey=scene.info['dims'][1],psizex=x_res,psizey=y_res,psizez=z_res,psizexu='um',psizeyu='um',psizezu='um',dimorder='TZYXC')

    # convert metadata to 7bit ascii
    metadata = metadata.encode('ascii',errors='ignore')
    return metadata

#create_metadata(img_list[2])


def get_array(scene,template_channel):
    # get the array of the current scene
    frames = []
    n_slices = scene.nz
    for i in range(n_slices):
        # get the frame
        frame = scene.get_frame(t=0,c=template_channel,z=i)
        # convert frame to numpy array
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)
    return frames

def get_template(scene,template_channel):
    n_channels = scene.info['channels']
    template = []
    # if nchannels > 1, select channel to use as template
    if n_channels > 1:
        template = get_array(scene,template_channel)
    else:
        template = get_array(scene,0)
    return template

# get_template(img_list[1],template_channel).shape

def get_binary_image(template):

    # push to GPU
    GPU_image = cle.push(template)
    # blur the image to remove noise
    GPU_image = cle.gaussian_blur(GPU_image, sigma_x=25, sigma_y=25, sigma_z=0)
    # binarize using otsu thresholding
    label_image = cle.threshold_otsu(GPU_image)
    # pull back to CPU
    label_image = cle.pull(label_image)
    # set all non-zero values to 255
    label_image[label_image != 0] = 255

    return label_image

# get_binary_image(get_template(img_list[1],template_channel)).shape

def find_largest_rectangle(bounding_rectangles):
    # Calculate the area of each bounding rectangle
    areas = [rect[2] * rect[3] for rect in bounding_rectangles]
    
    # Find the index of the bounding rectangle with the largest area
    largest_index = areas.index(max(areas))
    
    # Get the largest bounding rectangle
    largest_rectangle = bounding_rectangles[largest_index]
    
    return largest_rectangle


def get_bounding_rect(binary_image):

    bounding_rects = []
    for z in range(binary_image.shape[0]):
        # Find contours on the binary slice
        contours, hierarchy = cv2.findContours(binary_image[z, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if there is no contour, do not append the slice
        if len(contours) == 0:
            continue
        else:
            # pick the largest contour
            cnt = max(contours, key=cv2.contourArea)
        # Get bounding rectangle of largest contour
        x, y, w, h = cv2.boundingRect(cnt)

        bounding_rects.append([x, y, w, h])

    bounding_rect = find_largest_rectangle(bounding_rects)

    return bounding_rect

# get_bounding_rect(get_binary_image(get_template(img_list[1],template_channel)))


def crop_multichannel_stack(multichannel_stack,bounding_rect):
    # add same padding on all sides
    padding = 100
    bounding_rect = [bounding_rect[0] - padding, bounding_rect[1] - padding, bounding_rect[2] + 2*padding, bounding_rect[3] + 2*padding]

    # Crop multichannel_stack slicewise along the bounding rectangle
    cropped_frames = [multichannel_stack[:, z, bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]] for z in range(multichannel_stack.shape[1])]
    # stack the frames
    cropped_array = np.stack(cropped_frames, axis=1)
    # return the cropped image
    # add an extra dim for time
    cropped_array = np.expand_dims(cropped_array, axis=2)
    # reorder the axes from CZTXY to TZYXC 
    cropped_array = np.moveaxis(cropped_array, [0, 1, 2, 3, 4], [4, 1, 0, 3, 2])

    return cropped_array


# crop_multichannel_stack(scene_to_stack(img_list[1]),get_bounding_rect(get_binary_image(get_template(img_list[1],template_channel)))).shape


# def crop_binary_image(binary_image,bounding_rect):
#     # crop the binary image, too
#     cropped_binary_image = binary_image[:, bounding_rect[1]:bounding_rect[1] + bounding_rect[3], bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
#     return cropped_binary_image


def save_image(image,metadata,path):
    # save the cropped image stack
    tf.imwrite(path, image,description=metadata)


def process_scene(scene,template_channel):
    # get the template
    template = get_template(scene,template_channel)
    # get the binary image
    binary_image = get_binary_image(template)
    # get the bounding rectangle
    bounding_rect = get_bounding_rect(binary_image)
    # crop the multichannel stack
    cropped_multichannel_stack = crop_multichannel_stack(scene_to_stack(scene),bounding_rect)
    # crop the binary image
    #cropped_binary_image = crop_binary_image(binary_image,bounding_rect)
    # create metadata
    metadata = create_metadata(scene)
    position = scene.info['name']
    # replace "/" with "_" in position
    position = position.replace("/","_")
    # save the cropped multichannel stack
    save_image(cropped_multichannel_stack,metadata,args.lif.split(".")[0] +"_{scene}_cropped.tiff".format(scene=position))
    # save the cropped binary image
    #save_image(cropped_binary_image,metadata,"/media/geffjoldblum/DATA/ImagesMarwa/20230821_ehd2_laser_abl_02_{scene}_label_cropped.tiff".format(scene=position))
    print("Processed {scene}".format(scene=position))


#process_scene(img_list[1],template_channel)



for scene in img_list:
    process_scene(scene,template_channel)


